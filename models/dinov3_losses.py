# Adapted from facebookresearch/dinov3 (loss/gram_loss.py, loss/koleo_loss.py), bản không phân tán.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """MSE giữa ma trận tương quan patch (Gram) của student và target (anchor).
    feats: (B, N, D) -- Gram tính trong từng ảnh (img_level=True của bản gốc)."""

    def __init__(self, apply_norm: bool = True, remove_neg: bool = False):
        super().__init__()
        self.apply_norm, self.remove_neg = apply_norm, remove_neg

    def forward(self, student_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        assert student_feats.ndim == 3 and target_feats.ndim == 3
        s, t = student_feats.float(), target_feats.float().detach()
        if self.apply_norm:
            s, t = F.normalize(s, dim=-1), F.normalize(t, dim=-1)
        s_sim = s @ s.transpose(-1, -2)
        t_sim = t @ t.transpose(-1, -2)
        if self.remove_neg:
            s_sim, t_sim = s_sim.clamp(min=0), t_sim.clamp(min=0)
        return F.mse_loss(s_sim, t_sim)


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko (Sablayrolles et al. 2018): đẩy mỗi vector xa láng giềng gần nhất. x: (B, D)."""

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.normalize(x.float(), p=2, dim=-1, eps=eps)
            n = x.shape[0]
            with torch.no_grad():
                dots = x @ x.t()
                dots.view(-1)[:: n + 1].fill_(-1)  # loại chính nó
                idx = dots.argmax(dim=1)
            d = F.pairwise_distance(x, x[idx], p=2, eps=1e-8)
            return -torch.log(d + eps).mean()
