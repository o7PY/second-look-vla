"""Scoring a model on a split and producing the full metric set (SPEC §7)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from second_look import metrics as M
from second_look.dataset import PairDataset, make_loader, patch_masks, pixel_arm, to_float
from second_look.model import DeltaDINO, paint_arm_pixels


DEVICE = os.environ.get("SL_DEVICE", "cuda")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "none"


@torch.no_grad()
def score_model(model: DeltaDINO, split: str, batch: int = 64, workers: int = 4, include_gray: bool = True,
                device: str = DEVICE, ds: PairDataset | None = None, limit: int | None = None) -> dict:
    """Image logits + patch logits for every pair of a split (gray included by default). Returns a score dict."""
    model.eval()
    ds = ds or PairDataset(split, include_gray=include_gray)
    if limit:
        ds.rows = ds.rows[:limit]
    dl = make_loader(ds, batch, train=False, workers=workers)
    S, P, A, C, R = [], [], [], [], []
    for ref, st, mk, row in dl:
        ref, st, mk = ref.to(device, non_blocking=True), st.to(device, non_blocking=True), mk.to(device, non_blocking=True)
        ref, st, mk = to_float(ref, st, mk)
        arm, chg = patch_masks(mk)
        arm_in, arm = model.apply_static(arm), model.display_mask(arm)       # fed to the model / saved for pointing metrics and strata
        if model.mask_mode == "pixel":
            a_r, a_s = pixel_arm(mk)
            ref, st = paint_arm_pixels(ref, a_r), paint_arm_pixels(st, a_s)
        with torch.autocast(device, dtype=torch.bfloat16):
            s, z = model(ref, st, arm_in)
        S.append(s.float().cpu()), P.append(z.float().cpu()), A.append(arm.cpu()), C.append(chg.cpu()), R.append(row)
    return {"score": torch.cat(S).numpy(), "pmap": torch.cat(P).numpy(), "arm": torch.cat(A).numpy(),
            "chg": torch.cat(C).numpy(), "row": torch.cat(R).numpy()}


@torch.no_grad()
def latency_ms(model: DeltaDINO, n: int = 100, device: str = DEVICE) -> float:
    """ms per pair at batch 1 (includes the arm-mask path, excludes data loading)."""
    model.eval()
    ref = torch.rand(1, 3, 224, 224, device=device)
    arm = torch.zeros(1, 16, 16, dtype=torch.bool, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    with torch.autocast(device, dtype=torch.bfloat16):
        for _ in range(10):
            model(ref, ref, arm)
        sync()
        t = time.time()
        for _ in range(n):
            model(ref, ref, arm)
        sync()
    return (time.time() - t) / n * 1000


def load_model(ckpt_path: str, device: str = DEVICE) -> tuple[DeltaDINO, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ck["config"]["model"]
    model = DeltaDINO(lora_rank=mc["lora_rank"], lora_alpha=mc.get("lora_alpha", 16), lora_dropout=mc.get("lora_dropout", 0.05),
                      mask_mode=mc.get("mask_mode", "feature"), use_static=mc.get("use_static", False))
    model.load_trainable(ck["state"])
    return model.to(device), ck


def full_report(meta, sc: dict, thr: float, n_boot: int = 1000) -> dict:
    """All SPEC §7.1 numbers for one split, given scores and a frozen threshold."""
    ng = ~meta["gray"].to_numpy(bool)
    y, s = meta["label"].to_numpy()[ng], sc["score"][ng]
    rep = {"image": M.image_metrics(y, s, thr)}
    for stat in ("auroc", "ap", "f1", "precision", "recall"):
        rep["image"][f"{stat}_ci"] = M.bootstrap_ci(meta[ng].reset_index(drop=True), y, s, thr, stat, n_boot)
    sub = {k: v[ng] for k, v in sc.items()}
    rep["patch"] = M.patch_metrics(sub, y)
    rep["hard_neg"] = M.hard_negative_far(meta, sc["score"], thr)
    rep["big_change_recall"] = M.big_change_recall(meta, sc["score"], thr)
    rep["gray"] = M.gray_report(meta, sc["score"], thr)
    return rep


def val_threshold(method: str) -> float:
    """F1-maximising threshold on non-gray val pairs (SPEC §7.1); frozen afterwards."""
    meta, sc = M.load_scores(method, "val")
    ng = ~meta["gray"].to_numpy(bool)
    return M.best_f1_threshold(meta["label"].to_numpy()[ng], sc["score"][ng])


def report_from_scores(method: str, split: str, thr: float, n_boot: int = 1000) -> dict:
    meta, sc = M.load_scores(method, split)
    rep = full_report(meta, sc, thr, n_boot)
    rep["slices"] = {by: M.slice_table(meta, sc["score"], thr, by).to_dict("records") for by in ("source", "gen", "task", "magnitude", "delta")}
    rep["per_object"] = M.per_object(meta, sc["score"], thr).to_dict("records")
    rep["threshold"] = thr
    return rep


def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
