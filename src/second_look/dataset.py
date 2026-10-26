"""Torch dataset over the pair shards (SPEC §4.5, §5.1).

The npz shards are compressed, so random access would re-inflate ~150 MB per read. ``build_cache`` therefore
makes an uncompressed, memory-mapped copy per split (images uint8, masks bit-packed) that the OS pages in and
out; RAM stays flat. Augmentation (identical geometry + colour on both frames and all masks, independent noise)
runs on the GPU in ``augment``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from second_look.scene import ROOT
from second_look.sealed import check_sealed

PAIRS = Path(os.environ.get("SL_PAIRS", ROOT / "data" / "pairs"))
CACHE = Path(os.environ.get("SL_CACHE", ROOT / "data" / "cache"))
PIN = os.environ.get("SL_DEVICE", "cuda") == "cuda"
MASK_KEYS = ("arm_ref", "arm_set", "chg_ref", "chg_set")
PX = 224 * 224
PACKED = PX // 8
SPLITS = ("train", "val", "test", "test_heldout_tasks", "test_policy")


def build_cache(split: str, pairs_dir: Path = PAIRS, cache_dir: Path = CACHE, force: bool = False) -> Path:
    check_sealed(split)
    out = cache_dir / split
    if (out / "DONE").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(pairs_dir / split / "meta.parquet")
    n = len(meta)
    ref = np.lib.format.open_memmap(out / "ref.npy", "w+", np.uint8, (n, 224, 224, 3))
    st = np.lib.format.open_memmap(out / "set.npy", "w+", np.uint8, (n, 224, 224, 3))
    msk = np.lib.format.open_memmap(out / "masks.npy", "w+", np.uint8, (n, 4, PACKED))
    pos = 0
    for shard in sorted((pairs_dir / split).glob("shard-*.npz")):
        with np.load(shard) as z:
            r, s = z["ref"], z["set"]
            k = len(r)
            ref[pos:pos + k], st[pos:pos + k] = r, s
            for j, key in enumerate(MASK_KEYS):
                msk[pos:pos + k, j] = np.packbits(z[key].reshape(k, -1), axis=1)
            pos += k
        ref.flush(), st.flush(), msk.flush()
    assert pos == n, (pos, n)
    meta.to_parquet(out / "meta.parquet")
    (out / "DONE").write_text(str(n))
    return out


class PairDataset(torch.utils.data.Dataset):
    """Items: ref, set (3,224,224 uint8), masks (4,224,224 uint8: arm_ref, arm_set, chg_ref, chg_set), row index."""

    def __init__(self, split: str, include_gray: bool = False, cache_dir: Path = CACHE, build: bool = True, labels: str = "v1"):
        check_sealed(split)
        d = build_cache(split) if build else cache_dir / split
        self.dir, self.split = d, split
        self.meta = pd.read_parquet(d / "meta.parquet").reset_index(drop=True)
        if labels == "v2":                       # Part 1.5 label_v2 (symmetric-object tilt), data/pairs/<split>/labels_v2.parquet; v1 stays on disk
            uid = (self.meta.pair_id + "#" + self.meta.groupby("pair_id").cumcount().astype(str)).values
            lv = pd.read_parquet(PAIRS / split / "labels_v2.parquet").set_index("uid").loc[uid]
            self.meta["label"], self.meta["gray"], self.meta["max_rot_deg"] = lv.label_v2.values, lv.gray_v2.values.astype(bool), lv.max_rot_v2.values
        elif labels != "v1":
            raise ValueError(labels)
        self.ref = self.set = self.msk = None
        keep = np.ones(len(self.meta), bool) if include_gray else ~self.meta["gray"].to_numpy(bool)
        self.rows = np.flatnonzero(keep)

    def _open(self):
        if self.ref is None:
            self.ref = np.load(self.dir / "ref.npy", mmap_mode="r")
            self.set = np.load(self.dir / "set.npy", mmap_mode="r")
            self.msk = np.load(self.dir / "masks.npy", mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        self._open()
        r = int(self.rows[i])
        m = np.unpackbits(np.asarray(self.msk[r]), axis=1).reshape(4, 224, 224)
        return (torch.from_numpy(np.array(self.ref[r])).permute(2, 0, 1),
                torch.from_numpy(np.array(self.set[r])).permute(2, 0, 1),
                torch.from_numpy(m), r)

    def labels(self) -> np.ndarray:
        return self.meta["label"].to_numpy()[self.rows]


def balanced_weights(ds: PairDataset) -> np.ndarray:
    """Sampling weights so every (label x source x generator) cell is drawn equally often (SPEC §5.2)."""
    m = ds.meta.iloc[ds.rows]
    key = m["label"].astype(str) + "|" + m["source"] + "|" + m["gen"]
    cnt = key.map(key.value_counts())
    w = 1.0 / cnt.to_numpy(float)
    return w / w.sum()


def make_loader(ds: PairDataset, batch: int, train: bool, workers: int = 4, seed: int = 0, n_draws: int | None = None):
    if train:
        g = torch.Generator().manual_seed(seed)
        sampler = torch.utils.data.WeightedRandomSampler(balanced_weights(ds), n_draws or len(ds), replacement=True, generator=g)
        return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers, drop_last=True,
                                           pin_memory=PIN, persistent_workers=workers > 0)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=PIN)


# ---- GPU-side augmentation and patch masks -------------------------------------------------------
def _grid(theta, size=224):
    return F.affine_grid(theta, (theta.shape[0], 1, size, size), align_corners=False)


@torch.no_grad()
def augment(ref: torch.Tensor, st: torch.Tensor, masks: torch.Tensor, gen: torch.Generator | None = None):
    """ref, st: (B,3,H,W) uint8 on GPU; masks: (B,4,H,W) uint8. Returns float images in [0,1] and float masks.

    Identical random-resized-crop (scale .85-1, aspect .95-1.05), h-flip p=.5, brightness/contrast ±.1 for the pair
    and its masks; independent Gaussian noise sigma=2/255 per frame."""
    B, dev = ref.shape[0], ref.device
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, device=dev, generator=gen)
    area, asp = u(0.85, 1.0, B), u(0.95, 1.05, B)
    cw, ch = torch.sqrt(area * asp).clamp(max=1.0), torch.sqrt(area / asp).clamp(max=1.0)
    cx, cy = u(-1, 1, B) * (1 - cw), u(-1, 1, B) * (1 - ch)
    flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = cw * flip, cx, ch, cy
    grid = _grid(theta, ref.shape[-1])
    img = F.grid_sample(torch.cat([ref, st], 0).float() / 255.0, torch.cat([grid, grid], 0),
                        mode="bilinear", padding_mode="border", align_corners=False)
    mk = F.grid_sample(masks.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    a, b = img[:B], img[B:]
    br, ct = u(0.9, 1.1, B).view(B, 1, 1, 1), u(0.9, 1.1, B).view(B, 1, 1, 1)
    def jit(x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - mean) * ct + mean) * br
    a, b = jit(a), jit(b)
    a = a + torch.randn(a.shape, device=dev, generator=gen) * (2 / 255)
    b = b + torch.randn(b.shape, device=dev, generator=gen) * (2 / 255)
    return a.clamp(0, 1), b.clamp(0, 1), mk


def to_float(ref: torch.Tensor, st: torch.Tensor, masks: torch.Tensor):
    return ref.float() / 255.0, st.float() / 255.0, masks.float()


def patch_masks(masks: torch.Tensor, frac: float = 0.10):
    """masks (B,4,224,224) float -> (arm (B,16,16) bool, chg (B,16,16) bool) per SPEC §3.4."""
    cov = F.avg_pool2d(masks, 14) >= frac                          # (B,4,16,16)
    arm = (cov[:, 0] | cov[:, 1]).float().unsqueeze(1)
    arm = F.max_pool2d(arm, 3, stride=1, padding=1).squeeze(1) > 0   # dilate by 1 patch
    chg = cov[:, 2] | cov[:, 3]
    return arm, chg


def pixel_arm(masks: torch.Tensor):
    """(arm_ref, arm_set) pixel masks (B,224,224) bool, for the pixel-level masking ablation."""
    return masks[:, 0] > 0.5, masks[:, 1] > 0.5
