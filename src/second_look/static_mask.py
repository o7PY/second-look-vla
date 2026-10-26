"""C8: patches in which a changed object never appears in the TRAIN split (both frames), from the memory-mapped cache, streamed in chunks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from second_look.dataset import CACHE, PACKED, PairDataset
from second_look.scene import ROOT


def compute_static(force: bool = False, frac: float = 0.10) -> np.ndarray:
    """(16,16) bool; True = no train pair ever has >= ``frac`` of the patch covered by a changed object (ref or set)."""
    f = CACHE / "static_mask_train.npy"
    if f.exists() and not force:
        return np.load(f)
    d = PairDataset("train", include_gray=True).dir
    msk = np.load(d / "masks.npy", mmap_mode="r")
    count = np.zeros((16, 16), np.int64)
    for s in range(0, len(msk), 1000):
        m = np.unpackbits(np.asarray(msk[s:s + 1000]), axis=2).reshape(-1, 4, 224, 224)
        chg = torch.from_numpy(m[:, 2:4].max(axis=1).astype(np.float32))
        count += (F.avg_pool2d(chg[:, None], 14).squeeze(1) >= frac).sum(0).numpy()
    static = count == 0
    np.save(f, static)
    (ROOT / "results/part1/p15").mkdir(exist_ok=True)
    json.dump({"static_patches": int(static.sum()), "never_changed_rows_cols": [[int(a), int(b)] for a, b in zip(*np.nonzero(static))],
               "count_grid": count.tolist()}, open(ROOT / "results/part1/p15/static_mask.json", "w"))
    return static


def load_static_file(path: str) -> np.ndarray:
    """(16,16) bool from a json written by the before-Part-2 R2 step (`{"mask": [[...16x16 0/1...]]}`) or an .npy."""
    path = str(path)
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    if p.suffix == ".npy":
        return np.load(p).astype(bool)
    m = np.array(json.load(open(p))["mask"], dtype=bool)
    assert m.shape == (16, 16), m.shape
    return m
