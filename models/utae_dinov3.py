import copy
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_base import build_dinov3
from .utae_paps_models.utae import UTAE


class DinoV3Backbone(nn.Module):
    def __init__(self, in_chans, arch="dinov3_vits16", weights="LVD1689M", pretrained=True,
                 out_layers: Optional[Sequence[int]] = None, input_scale: float = 2.0, n_out: int = 2):
        super().__init__()
        self.vit = build_dinov3(arch, weights, pretrained)
        self.vit.adapt_in_chans(in_chans)
        self.patch_size, self.embed_dim = self.vit.patch_size, self.vit.embed_dim
        n = self.vit.n_blocks
        default = [n - 1] if n_out == 1 else [n // 2 - 1, n - 1]  # 1 stage: chỉ layer cuối; 2 stage: giữa + cuối
        self.out_layers = list(out_layers) if out_layers is not None else default
        self.input_scale = input_scale
        self._frozen = []  # list Parameter (không đăng ký vào module)

    # ---- đóng/mở băng: requires_grad luôn True (an toàn cho DDP), grad của tham số "đóng băng" bị bỏ sau backward ----
    def set_train_mode(self, mode: str = "patch_only", last_n_blocks: int = 4):
        """patch_only: chỉ patch_embed | partial: + n block cuối + norm | full: tất cả"""
        assert mode in ("patch_only", "partial", "full")
        train_ids = {id(p) for p in self.vit.patch_embed.parameters()}
        if mode == "partial":
            for m in [*self.vit.blocks[-last_n_blocks:], self.vit.norm]:
                train_ids |= {id(p) for p in m.parameters()}
        elif mode == "full":
            train_ids |= {id(p) for p in self.vit.parameters()}
        self._frozen = [p for p in self.vit.parameters() if id(p) not in train_ids]

    def drop_frozen_grads(self):
        for p in self._frozen:
            p.grad = None

    def forward_flat(self, xf: torch.Tensor, vit: Optional[nn.Module] = None):
        """xf (N,C,H,W) -> (list feature map (N,D,gh,gw), tokens cuối (N,P,D))"""
        vit = vit or self.vit
        p = self.patch_size
        H, W = xf.shape[-2:]
        h = max(p, round(H * self.input_scale / p) * p)
        w = max(p, round(W * self.input_scale / p) * p)
        xf = F.interpolate(xf, size=(h, w), mode="bilinear", align_corners=False)
        outs = vit.forward_intermediate(xf, self.out_layers)
        gh, gw = h // p, w // p
        feats = [t.transpose(1, 2).reshape(t.shape[0], -1, gh, gw) for t, _ in outs]
        return feats, outs[-1][0]


class DinoFusion(nn.Module):
    """hybrid: out = conv_feat + gate * mix([conv_feat, proj(dino)]),  gate khởi tạo 0 (từng kênh)
    -> lúc đầu mô hình == UTAE gốc; DINOv3 chỉ đóng góp khi gradient thấy có ích.
    replace: out = proj(dino) (không có conv)."""

    def __init__(self, c_conv, c_dino, mode="hybrid", groups=4):
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(nn.Conv2d(c_dino, c_conv, 1, bias=False), nn.GroupNorm(groups, c_conv), nn.GELU())
        if mode == "hybrid":
            self.mix = nn.Sequential(nn.Conv2d(2 * c_conv, c_conv, 3, padding=1, bias=False), nn.GroupNorm(groups, c_conv))
            self.gate = nn.Parameter(torch.zeros(1, c_conv, 1, 1))

    def forward(self, dino_feat, size, B, T, conv_feat=None):
        d = F.interpolate(self.proj(dino_feat), size=tuple(size), mode="bilinear", align_corners=False)
        if self.mode == "hybrid":
            c = conv_feat.flatten(0, 1)
            d = c + self.gate * self.mix(torch.cat([c, d], dim=1))
        return d.view(B, T, *d.shape[1:])


class UTAEDinoV3(UTAE):
    def __init__(self, input_dim, dino_arch="dinov3_vits16", dino_weights="LVD1689M", dino_pretrained=True,
                 dino_input_scale=2.0, fusion_mode="hybrid", fusion_stages: Sequence[int] = (3,), **utae_kwargs):
        super().__init__(input_dim=input_dim, **utae_kwargs)
        assert fusion_mode in ("hybrid", "replace")
        self.fusion_mode, self.fusion_stages = fusion_mode, list(fusion_stages)
        assert len(self.fusion_stages) in (1, 2)
        self.dino = DinoV3Backbone(input_dim, dino_arch, dino_weights, dino_pretrained,
                                   input_scale=dino_input_scale, n_out=len(self.fusion_stages))
        assert len(self.dino.out_layers) == len(self.fusion_stages)
        self.fusers = nn.ModuleDict(
            {str(s): DinoFusion(self.encoder_widths[s], self.dino.embed_dim, fusion_mode) for s in self.fusion_stages})
        if fusion_mode == "replace":
            for s in self.fusion_stages:
                self.down_blocks[s - 1] = nn.Identity()
        self.collect_aux = True
        self.aux: Optional[Dict[str, torch.Tensor]] = None
        self._anchor_vit = None  # ẩn khỏi state_dict / parameters()

    # ---- Gram anchor: snapshot đóng băng của ViT TẠI THỜI ĐIỂM bắt đầu fine-tune ----
    def has_anchor(self):
        return self._anchor_vit is not None

    @torch.no_grad()
    def make_anchor(self):
        a = copy.deepcopy(self.dino.vit).eval()
        for p in a.parameters():
            p.requires_grad = False
        object.__setattr__(self, "_anchor_vit", a)  # bypass đăng ký nn.Module -> không vào checkpoint

    def pop_aux(self):
        aux, self.aux = self.aux, None
        return aux

    def forward(self, input, batch_positions=None, return_att=False):
        B, T, C, H, W = input.shape
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # B x T

        xf = input.flatten(0, 1)
        dino_feats, tokens = self.dino.forward_flat(xf)
        dino_maps = dict(zip(self.fusion_stages, dino_feats))

        feature_maps = [self.in_conv.smart_forward(input)]
        for i in range(self.n_stages - 1):
            s, prev = i + 1, feature_maps[-1]
            if s in dino_maps and self.fusion_mode == "replace":
                size = (prev.shape[-2] // 2, prev.shape[-1] // 2)  # k=4,s=2,p=1 -> floor(h/2)
                out = self.fusers[str(s)](dino_maps[s], size, B, T)
            else:
                out = self.down_blocks[i].smart_forward(prev)
                if s in dino_maps:
                    out = self.fusers[str(s)](dino_maps[s], out.shape[-2:], B, T, conv_feat=out)
            feature_maps.append(out)

        out, att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att)
            out = self.up_blocks[i](out, skip)
        out = self.out_conv(out)

        if self.training and self.collect_aux:
            valid = ~pad_mask.flatten()
            w = (~pad_mask).float().unsqueeze(-1)  # B,T,1
            pooled = (tokens.mean(1).view(B, T, -1) * w).sum(1) / w.sum(1).clamp(min=1.0)  # B,D
            aux = {"student_tokens": tokens[valid], "pooled": pooled, "anchor_tokens": None}
            if self.has_anchor() and valid.any():
                with torch.no_grad():
                    _, aux["anchor_tokens"] = self.dino.forward_flat(xf[valid], vit=self._anchor_vit)
            self.aux = aux
        return (out, att) if return_att else out
