"""Metrics and slices shared by DeltaDINO and every baseline (SPEC §7.1).

A "score file" is an npz per (method, split): ``score`` (N,) higher = more changed, ``pmap`` (N,16,16) patch
scores, ``arm`` and ``chg`` (N,16,16) bool patch masks, ``row`` (N,) row indices into the split's meta.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import os
from pathlib import Path

from second_look.scene import ROOT

SCORES = Path(os.environ.get("SL_SCORES", ROOT / "results" / "part1" / "scores"))
CACHE_DIR = Path(os.environ.get("SL_CACHE", ROOT / "data" / "cache"))


def load_scores(method: str, split: str):
    z = np.load(SCORES / method / f"{split.replace('/', '__')}.npz")
    meta = pd.read_parquet(CACHE_DIR / split / "meta.parquet").iloc[z["row"]].reset_index(drop=True)
    return meta, {k: z[k] for k in z.files}


def save_scores(method: str, split: str, **arrays) -> None:
    d = SCORES / method
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{split.replace('/', '__')}.npz", **arrays)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def prf(y, s, thr) -> dict:
    pred = s >= thr
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)}


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def image_metrics(y, s, thr) -> dict:
    ok = len(np.unique(y)) == 2
    return {"n": int(len(y)), "pos": int(y.sum()), "auroc": auroc(y, s),
            "ap": float(average_precision_score(y, s)) if ok else float("nan"), **prf(y, s, thr)}


def episode_ids(meta: pd.DataFrame) -> np.ndarray:
    return (meta["task"].astype(int) * 1000 + meta["demo"].astype(int)).to_numpy()


def bootstrap_ci(meta, y, s, thr, stat="auroc", n_boot: int = 1000, seed: int = 0):
    """95% CI by resampling episodes (clusters) with replacement."""
    ep = episode_ids(meta)
    uniq, inv = np.unique(ep, return_inverse=True)
    idx_by = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(len(uniq), size=len(uniq))
        ii = np.concatenate([idx_by[p] for p in pick])
        yy, ss = y[ii], s[ii]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(auroc(yy, ss) if stat == "auroc" else (average_precision_score(yy, ss) if stat == "ap" else prf(yy, ss, thr)[stat]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def patch_metrics(sc: dict, y: np.ndarray) -> dict:
    """Patch-level AUROC on non-arm patches of positive pairs; pointing accuracy = top non-arm patch is a changed patch."""
    pm, arm, chg = sc["pmap"], sc["arm"], sc["chg"]
    pos = y == 1
    keep = ~arm[pos]
    yt, ys = chg[pos][keep], pm[pos][keep]
    pa = auroc(yt.astype(int), ys) if 0 < yt.sum() < len(yt) else float("nan")
    masked = np.where(arm[pos], -np.inf, pm[pos]).reshape(pos.sum(), -1)
    top = masked.argmax(1)
    hit = chg[pos].reshape(pos.sum(), -1)[np.arange(pos.sum()), top]
    has = chg[pos].reshape(pos.sum(), -1).any(1)
    return {"patch_auroc": pa, "pointing_acc": float(hit[has].mean()) if has.any() else float("nan"), "n_pos_with_chg_patch": int(has.sum())}


def magnitude_bin(meta: pd.DataFrame) -> pd.Series:
    tr, ro = meta["max_trans_cm"], meta["max_rot_deg"]
    b = np.full(len(meta), "none", dtype=object)
    b[(ro >= 5) & (ro <= 15)] = "rot 5-15deg"
    b[ro > 15] = "rot >15deg"
    b[(tr > 1) & (tr <= 2)] = "trans 1-2cm"
    b[(tr > 2) & (tr <= 5)] = "trans 2-5cm"
    b[tr > 5] = "trans >5cm"
    return pd.Series(b, index=meta.index)


def slice_table(meta: pd.DataFrame, s: np.ndarray, thr: float, by: str) -> pd.DataFrame:
    """Per-slice metrics on non-gray pairs. For slices with one class only, AUROC is NaN and we report the
    positive-call rate (= recall for all-positive slices, false-alarm rate for all-negative slices)."""
    rows = []
    ng = ~meta["gray"].to_numpy(bool)
    col = meta[by] if by != "magnitude" else magnitude_bin(meta)
    for key in sorted(col.unique(), key=str):
        m = ng & (col == key).to_numpy()
        if m.sum() == 0:
            continue
        y = meta["label"].to_numpy()[m]
        rows.append({by: key, "n": int(m.sum()), "pos": int(y.sum()), "call_rate": float((s[m] >= thr).mean()),
                     "auroc": auroc(y, s[m])})
    return pd.DataFrame(rows)


def per_object(meta: pd.DataFrame, s: np.ndarray, thr: float) -> pd.DataFrame:
    """Recall by which object changed (positive, single-object-changed pairs only)."""
    ng = (~meta["gray"].to_numpy(bool)) & (meta["label"].to_numpy() == 1)
    rows = []
    items = meta["changed_items"].fillna("").to_numpy()
    for name in sorted({i for it in items[ng] for i in it.split(",") if i}):
        m = ng & np.array([name in it.split(",") for it in items])
        rows.append({"object": name, "n": int(m.sum()), "recall": float((s[m] >= thr).mean())})
    return pd.DataFrame(rows)


def gray_report(meta: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    g = meta["gray"].to_numpy(bool)
    return {"n": int(g.sum()), "call_rate": float((s[g] >= thr).mean()) if g.any() else float("nan")}


def hard_negative_far(meta: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    out = {}
    ng = ~meta["gray"].to_numpy(bool)
    for g in ("H1", "H2"):
        m = ng & (meta["gen"] == g).to_numpy()
        out[g] = {"n": int(m.sum()), "false_alarm": float((s[m] >= thr).mean()) if m.any() else float("nan")}
    m = ng & meta["gen"].isin(["H1", "H2"]).to_numpy()
    out["H1+H2"] = {"n": int(m.sum()), "false_alarm": float((s[m] >= thr).mean()) if m.any() else float("nan")}
    return out


def big_change_recall(meta: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    """Recall on positives with translation > 2 cm or rotation > 15 deg (success criterion)."""
    m = (meta["label"].to_numpy() == 1) & ~meta["gray"].to_numpy(bool) & ((meta["max_trans_cm"] > 2) | (meta["max_rot_deg"] > 15)).to_numpy()
    return {"n": int(m.sum()), "recall": float((s[m] >= thr).mean()) if m.any() else float("nan")}
