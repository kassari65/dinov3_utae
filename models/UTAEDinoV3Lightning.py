from typing import Any, List, Optional

import torch
from segmentation_models_pytorch.losses import DiceLoss

from .BaseModel import BaseModel
from .dinov3_losses import GramLoss, KoLeoLoss
from .utae_dinov3 import UTAEDinoV3


class UTAEDinoV3Lightning(BaseModel):
    """
    L = L_seg (BaseModel: BCE/Focal/...)                         loss chính
      + w_dice  * L_dice                                         overlap vùng cháy
      + w_gram  * L_gram   (chỉ bật sau giai đoạn warmup)        giữ cấu trúc patch của ViT lúc bắt đầu fine-tune
      + w_koleo * L_koleo   (mặc định 0 = tắt)                    rải đều embedding mức sample
    Mỗi loss phụ đều là giả thuyết -> cần ablation (đặt trọng số = 0 để tắt).
    """

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        dino_arch: str = "dinov3_vits16",       # dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16
        dino_weights: str = "LVD1689M",         # LVD1689M | SAT493M (chỉ vitl16) | đường dẫn .pth
        dino_pretrained: bool = True,
        dino_input_scale: float = 2.0,
        fusion_mode: str = "hybrid",            # hybrid | replace
        fusion_stages: Optional[List[int]] = None,  # mặc định [3] (chỉ bottleneck); [2, 3] để fuse cả stage 2
        dino_warmup_mode: str = "patch_only",
        dino_unfreeze_epoch: int = 5,
        dino_finetune_mode: str = "partial",    # partial | full | patch_only
        dino_unfreeze_last_n: int = 4,
        dino_lr_mult: float = 0.01,             # LR các block ViT = lr_optimizer * mult
        patch_embed_lr_mult: float = 0.1,       # LR conv patch 40 kênh mới
        w_dice: float = 0.25,
        w_gram: float = 1.0,
        w_koleo: float = 0.0,
        use_doy: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=use_doy,
            *args,
            **kwargs,
        )
        self.model = UTAEDinoV3(
            input_dim=n_channels, dino_arch=dino_arch, dino_weights=dino_weights,
            dino_pretrained=dino_pretrained, dino_input_scale=dino_input_scale, fusion_mode=fusion_mode,
            fusion_stages=tuple(fusion_stages or [3]),
            encoder_widths=[64, 64, 64, 128], decoder_widths=[32, 32, 64, 128], out_conv=[32, 1],
            str_conv_k=4, str_conv_s=2, str_conv_p=1, agg_mode="att_group", encoder_norm="group",
            n_head=16, d_model=256, d_k=4, encoder=False, return_maps=False, pad_value=0, padding_mode="reflect",
        )
        self.dice_loss, self.gram_loss, self.koleo_loss = DiceLoss(mode="binary"), GramLoss(), KoLeoLoss()
        self.w_dice, self.w_gram, self.w_koleo = w_dice, w_gram, w_koleo
        self.warmup_mode, self.finetune_mode = dino_warmup_mode, dino_finetune_mode
        self.unfreeze_epoch, self.unfreeze_last_n = dino_unfreeze_epoch, dino_unfreeze_last_n
        self.dino_lr_mult, self.patch_lr_mult = dino_lr_mult, patch_embed_lr_mult
        # KHÔNG gọi save_hyperparameters() lần nữa: sẽ ghi đè pos_class_weight đã chuẩn hoá của BaseModel
        self.hparams.update(dino_arch=dino_arch, dino_weights=str(dino_weights), fusion_mode=fusion_mode, fusion_stages=list(fusion_stages or [3]),
                            w_dice=w_dice, w_gram=w_gram, w_koleo=w_koleo)
        self.model.dino.set_train_mode(dino_warmup_mode, dino_unfreeze_last_n)

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)

    # ------------------------------------------------------------------ optimizer
    def on_fit_start(self) -> None:
        """LightningCLI (block `optimizer:` trong YAML) ghi đè configure_optimizers() và tạo 1 param-group duy nhất
        với lr=0.01 cho TOÀN BỘ model. Ở đây tách ViT ra 2 group riêng với LR nhỏ hơn. Hook này chạy sau khi
        optimizer được tạo và trước khi khôi phục optimizer state từ checkpoint (nên resume vẫn khớp)."""
        super().on_fit_start()  # BaseModel in cấu hình loss
        opt = self.trainer.optimizers[0]
        if len(opt.param_groups) != 1:
            return
        g0 = opt.param_groups[0]
        patch = [p for p in self.model.dino.vit.patch_embed.parameters()]
        patch_ids = {id(p) for p in patch}
        vit = [p for p in self.model.dino.vit.parameters() if id(p) not in patch_ids]
        moved = patch_ids | {id(p) for p in vit}
        base_lr, wd = g0["lr"], g0.get("weight_decay", 0.0)
        g0["params"] = [p for p in g0["params"] if id(p) not in moved]
        opt.add_param_group({"params": vit, "lr": base_lr * self.dino_lr_mult, "weight_decay": wd})
        opt.add_param_group({"params": patch, "lr": base_lr * self.patch_lr_mult, "weight_decay": wd})
        self.print(f"[DINO] LR: base={base_lr:g}, ViT blocks={base_lr * self.dino_lr_mult:g}, "
                   f"patch_embed={base_lr * self.patch_lr_mult:g}")

    def on_train_epoch_start(self) -> None:
        finetuning = self.current_epoch >= self.unfreeze_epoch
        self.model.dino.set_train_mode(self.finetune_mode if finetuning else self.warmup_mode, self.unfreeze_last_n)
        # Gram anchor = snapshot ViT (đã gồm patch_embed 40 kênh vừa học xong) ngay khi bắt đầu fine-tune
        if finetuning and self.w_gram > 0 and not self.model.has_anchor():
            self.model.make_anchor()
            self.print(f"[DINO] Gram anchor snapshot tại epoch {self.current_epoch}")

    def on_after_backward(self) -> None:
        self.model.dino.drop_frozen_grads()

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)
        if self.hparams.crop_before_eval:
            y_hat, y = self.center_crop(y_hat, y)

        l_seg = self.compute_loss(y_hat, y)                                    # (1) loss chính
        total = l_seg
        zero = y_hat.new_zeros(())
        l_dice = l_gram = l_koleo = zero

        if self.w_dice > 0:                                                    # (2) Dice
            l_dice = self.dice_loss(y_hat, y.float())
            total = total + self.w_dice * l_dice

        aux = self.model.pop_aux()
        if aux is not None:
            if self.w_gram > 0 and aux["anchor_tokens"] is not None:           # (3) DINOv3 Gram loss
                l_gram = self.gram_loss(aux["student_tokens"], aux["anchor_tokens"])
                total = total + self.w_gram * l_gram
            if self.w_koleo > 0 and aux["pooled"].shape[0] >= 2:               # (4) DINOv3 KoLeo loss
                l_koleo = self.koleo_loss(aux["pooled"])
                total = total + self.w_koleo * l_koleo

        probs = torch.sigmoid(y_hat)
        self.train_f1(probs, y)
        self.log_dict({"train_loss_seg": l_seg.detach(), "train_loss_dice": l_dice.detach(),
                       "train_loss_gram": l_gram.detach(), "train_loss_koleo": l_koleo.detach()},
                      on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train_loss", total.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total
    # validation/test kế thừa BaseModel: val_loss = L_seg thuần -> so sánh công bằng với baseline
