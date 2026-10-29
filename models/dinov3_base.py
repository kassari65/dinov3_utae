# Adapted from facebookresearch/dinov3 (ViT part only, inference/fine-tuning subset).
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
"""
DINOv3 ViT tự chứa (không import gì từ repo `dinov3`).

Giữ NGUYÊN tên module / tham số / buffer so với bản gốc để load_state_dict(strict=True)
checkpoint chính thức của Meta. Đã lược bỏ những thứ không cần khi fine-tune:
  fp8, ConvNeXt, SwiGLU/RMSNorm (3 preset vits16/vitb16/vitl16 không dùng), stochastic depth,
  multi-crop/forward_list, các head hub (detector/segmentor/dinotxt).
Khác biệt có chủ ý: RoPE KHÔNG có augmentation shift/jitter/rescale (bản gốc bật rescale=2 khi
model.train() -> làm feature nhiễu khi fine-tune và lệch so với anchor ở chế độ eval).
"""
import math
import os
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"

# cấu hình lấy từ dinov3/hub/backbones.py
_PRESETS = {
    "dinov3_vits16": dict(embed_dim=384, depth=12, num_heads=6, hash={"LVD1689M": "08c60483"}),
    "dinov3_vitb16": dict(embed_dim=768, depth=12, num_heads=12, hash={"LVD1689M": "73cec8be"}),
    "dinov3_vitl16": dict(embed_dim=1024, depth=24, num_heads=16, hash={"LVD1689M": "8aa4cbdd", "SAT493M": "eadcf0ff"}),
}


# --------------------------------------------------------------------------- layers
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """(B,C,H,W) -> (B,H/p,W/p,D)"""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class RopePositionEmbedding(nn.Module):
    """RoPE axial, không học, tham số hoá bằng `base`."""

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        self.base = base
        self.D_head = embed_dim // num_heads
        self.register_buffer("periods", torch.empty(self.D_head // 4, dtype=torch.float32), persistent=True)
        self._init_weights()

    def _init_weights(self):
        e = 2 * torch.arange(self.D_head // 4, dtype=torch.float32) / (self.D_head // 2)
        self.periods.data.copy_(self.base ** e)

    def forward(self, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dd = {"device": self.periods.device, "dtype": torch.float32}
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)  # [HW,2]
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW,2,D//4]
        angles = angles.flatten(1, 2).tile(2)  # [HW, D]
        return torch.sin(angles), torch.cos(angles)


def _rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x, sin, cos):
    return (x * cos) + (_rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_features % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, x):
        b = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(x, self.weight, b)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_bias=True, mask_k_bias=False):
        super().__init__()
        self.num_heads = num_heads
        lin = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = lin(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    @staticmethod
    def _apply_rope(q, k, rope):
        q_dt, k_dt = q.dtype, k.dtype
        sin, cos = rope
        q, k = q.to(sin.dtype), k.to(sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]  # cls + storage tokens không có rope
        q = torch.cat((q[:, :, :prefix], _rope_apply(q[:, :, prefix:], sin, cos)), dim=-2)
        k = torch.cat((k[:, :, :prefix], _rope_apply(k[:, :, prefix:], sin, cos)), dim=-2)
        return q.to(q_dt), k.to(k_dt)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        return self.proj(x.reshape(B, N, C))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio, qkv_bias, proj_bias, ffn_bias, init_values, norm_eps, mask_k_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias, mask_k_bias)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * ffn_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


# --------------------------------------------------------------------------- model
class DinoVisionTransformer(nn.Module):
    def __init__(
        self, *, patch_size=16, in_chans=3, embed_dim, depth, num_heads, ffn_ratio=4.0,
        qkv_bias=True, proj_bias=True, ffn_bias=True, layerscale_init=1e-5, norm_eps=1e-5,
        n_storage_tokens=4, mask_k_bias=True, untie_global_and_local_cls_norm=False, rope_base=100.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads, base=rope_base)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(embed_dim, num_heads, ffn_ratio, qkv_bias, proj_bias, ffn_bias,
                                   layerscale_init, norm_eps, mask_k_bias)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        # chỉ dùng khi pretraining SSL; giữ lại để khớp state_dict của bản SAT493M
        self.local_cls_norm = nn.LayerNorm(embed_dim, eps=norm_eps) if untie_global_and_local_cls_norm else None
        if self.local_cls_norm is not None:
            for p in self.local_cls_norm.parameters():
                p.requires_grad = False  # không bao giờ nằm trong đồ thị -> tránh lỗi "unused parameters" của DDP
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))

    # ---- init (chỉ dùng khi pretrained=False) ----
    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if hasattr(m, "bias_mask"):
                    o = m.out_features
                    m.bias_mask.fill_(1)
                    m.bias_mask[o // 3 : 2 * o // 3].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, (LayerScale, PatchEmbed)):
                m.reset_parameters()

    def adapt_in_chans(self, in_chans: int):
        """3 kênh RGB -> C kênh: trung bình trọng số RGB, lặp C lần, nhân 3/C (heuristic của timm)."""
        pe = self.patch_embed
        old = pe.proj
        if old.in_channels == in_chans:
            return
        new = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1) * (3.0 / in_chans))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        pe.proj, pe.in_chans = new, in_chans

    def forward_intermediate(self, x: torch.Tensor, layers: Sequence[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B,C,H,W) -> [(patch_tokens (B,N,D), cls (B,D))] cho từng layer trong `layers` (đã qua self.norm).
        Dừng ngay tại layer lớn nhất (không tính các block phía sau)."""
        layers = sorted(set(layers))
        B = x.shape[0]
        x = self.patch_embed(x)
        _, h, w, _ = x.shape
        x = x.flatten(1, 2)
        cls = self.cls_token + 0 * self.mask_token  # giữ mask_token trong đồ thị (DDP)
        prefix = [cls.expand(B, -1, -1)]
        if self.n_storage_tokens > 0:
            prefix.append(self.storage_tokens.expand(B, -1, -1))
        x = torch.cat(prefix + [x], dim=1)
        rope = self.rope_embed(H=h, W=w)
        taken = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope)
            if i in layers:
                taken.append(x)
            if i == layers[-1]:
                break
        ns = self.n_storage_tokens
        out = []
        for t in taken:
            t = self.norm(t)
            out.append((t[:, ns + 1 :], t[:, 0]))
        return out


# --------------------------------------------------------------------------- builder / weights
def _weights_key(weights):
    if isinstance(weights, str) and weights.upper() in ("LVD1689M", "SAT493M"):
        return weights.upper()
    return None


def build_dinov3(arch: str = "dinov3_vits16", weights: str = "LVD1689M", pretrained: bool = True) -> DinoVisionTransformer:
    """weights: 'LVD1689M' | 'SAT493M' (chỉ vitl16) | đường dẫn file .pth | URL. Trả về model 3 kênh."""
    if arch not in _PRESETS:
        raise ValueError(f"arch phải thuộc {list(_PRESETS)}, nhận {arch!r}")
    cfg = _PRESETS[arch]
    key = _weights_key(weights)
    if pretrained and key is not None and key not in cfg["hash"]:
        raise ValueError(f"{arch} không có trọng số {key}; chỉ có {list(cfg['hash'])}")
    # bản SAT493M của ViT-L có thêm local_cls_norm (nhận biết qua tên/hash của file)
    sat = key == "SAT493M" or (key is None and isinstance(weights, str) and "eadcf0ff" in os.path.basename(weights))
    model = DinoVisionTransformer(
        patch_size=16, embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"],
        untie_global_and_local_cls_norm=bool(sat and arch == "dinov3_vitl16"),
    )
    if not pretrained:
        model.init_weights()
        return model
    if key is not None:
        url = f"{_DINOV3_BASE_URL}/{arch}/{arch}_pretrain_{key.lower()}-{cfg['hash'][key]}.pth"
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
    elif str(weights).startswith(("http://", "https://")):
        state = torch.hub.load_state_dict_from_url(str(weights), map_location="cpu", weights_only=True)
    else:
        state = torch.load(os.path.expanduser(str(weights)), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model
