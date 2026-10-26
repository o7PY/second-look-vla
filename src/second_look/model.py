"""DeltaDINO: Siamese, arm-masked, patch-level change detector on a LoRA-tuned DINOv2 ViT-S/14 (SPEC §1)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
GRID = 16
R_SMOOTH = 5.0


def smooth_max(z: torch.Tensor, keep: torch.Tensor, r: float = R_SMOOTH) -> torch.Tensor:
    """(1/r) log mean_{p in keep} exp(r z_p). z, keep: (B, 16, 16). Images with no kept patch get -10."""
    zf, kf = z.flatten(1).float(), keep.flatten(1)
    lse = torch.logsumexp((r * zf).masked_fill(~kf, float("-inf")), dim=1)
    n = kf.sum(1).clamp(min=1)
    s = (lse - torch.log(n.float())) / r
    return torch.where(kf.any(1), s, torch.full_like(s, -10.0))


class Head(nn.Module):
    def __init__(self, in_ch: int = 1153):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 1, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class DeltaDINO(nn.Module):
    def __init__(self, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 mask_mode: str = "feature", backbone: str = "facebook/dinov2-small", use_static: bool = False):
        """mask_mode: "feature" (default: arm patches zeroed + mask channel), "none" (no arm masking), "pixel" (arm pixels painted grey before the encoder)
        or "static" (candidate C: NO arm masking, but the static-region patches are masked at feature level; needs use_static=True).
        The two masks are independent: the arm mask is used by feature/pixel, the static mask by every mode with use_static=True except that
        "none" ignores both. lora_rank 0 = frozen backbone (baseline B3)."""
        super().__init__()
        from transformers import AutoModel
        assert mask_mode in ("feature", "none", "pixel", "static"), mask_mode
        assert mask_mode != "static" or use_static, "mask_mode 'static' needs use_static=True"
        self.mask_mode = mask_mode
        enc = AutoModel.from_pretrained(backbone)
        for p in enc.parameters():
            p.requires_grad = False
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                             target_modules=r".*encoder\.layer\.\d+\.attention\.(attention\.(query|key|value)|output\.dense)")
            enc = get_peft_model(enc, cfg)
        self.enc = enc
        self.has_lora = lora_rank > 0
        self.head = Head()
        # C8: patches (16x16) where a changed object never appears in the TRAIN split; treated like arm patches (zeroed, mask channel set,
        # excluded from pooling). Stored here, applied by the caller (train: warped with the augmentation; eval: ``apply_static``).
        self.use_static = use_static
        self.register_buffer("static", torch.zeros(GRID, GRID, dtype=torch.bool) if use_static else None)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    # --- helpers ---------------------------------------------------------------------------
    def normalise(self, x_uint8_or_float: torch.Tensor) -> torch.Tensor:
        x = x_uint8_or_float
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return (x - self.mean) / self.std

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) normalised -> (B,384,16,16) patch tokens (CLS discarded)."""
        h = self.enc(pixel_values=x).last_hidden_state[:, 1:]
        B, N, C = h.shape
        g = int(math.sqrt(N))
        return h.transpose(1, 2).reshape(B, C, g, g)

    def forward(self, ref: torch.Tensor, st: torch.Tensor, arm: torch.Tensor):
        """ref, st: (B,3,224,224) floats in [0,1] or uint8; arm: (B,16,16) bool (patch arm mask).
        Returns (image logit s (B,), patch logits z (B,16,16))."""
        B = ref.shape[0]
        a, b = self.normalise(ref), self.normalise(st)
        feats = self.features(torch.cat([a, b], 0))
        f_ref, f_set = feats[:B], feats[B:]
        m = arm.float().unsqueeze(1)
        if self.mask_mode == "none":
            m = torch.zeros_like(m)
            keep = torch.ones_like(arm)
        else:
            keep = ~arm
        x = torch.cat([f_ref, f_set, (f_set - f_ref).abs()], 1)
        if self.mask_mode in ("feature", "static"):          # feature-level zeroing (arm patches, or static patches in 'static' mode)
            x = x * (1 - m)
        x = torch.cat([x, m], 1)
        z = self.head(x)
        return smooth_max(z, keep), z

    # --- bookkeeping -----------------------------------------------------------------------
    def trainable_state(self) -> dict:
        """LoRA + head weights only (the backbone comes from HF)."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if k.startswith("head.") or "lora_" in k or k == "static"}

    def load_trainable(self, sd: dict) -> None:
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k.startswith("head.") or "lora_" in k]
        assert not bad and not unexpected, (bad, unexpected)

    def apply_static(self, arm: torch.Tensor) -> torch.Tensor:
        """The mask the model is FED, from the rendered arm patch mask (B,16,16) bool:
        'static' mode -> the static region only (the arm is ignored); a model with a static mask and arm masking -> arm OR static;
        no static mask -> arm unchanged ('none' ignores it inside forward). Idempotent."""
        if self.static is None:
            return arm
        st = self.static.to(arm.device).unsqueeze(0)
        return st.expand_as(arm).clone() if self.mask_mode == "static" else arm | st

    def display_mask(self, arm: torch.Tensor) -> torch.Tensor:
        """Mask used to EXCLUDE patches from pointing metrics and strata: rendered arm OR static region, whatever the model masks (comparable across candidates)."""
        return arm if self.static is None else arm | self.static.to(arm.device).unsqueeze(0)

    def param_counts(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters() if "lora_" in n)
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": tot, "trainable": tr, "lora": lora, "head": head}


def paint_arm_pixels(img: torch.Tensor, arm_px: torch.Tensor) -> torch.Tensor:
    """Pixel-level masking ablation: img (B,3,H,W) in [0,1], arm_px (B,H,W) bool -> arm painted mid-grey."""
    return torch.where(arm_px.unsqueeze(1), torch.full_like(img, 0.5), img)
