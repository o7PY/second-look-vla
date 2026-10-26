"""Part 1.5 helpers: score loading, row-wise image access, sheet selection (SPEC_part1_5_checks.md).

Everything streams: images come from the memory-mapped split cache one row at a time; scores are the saved
per-pair arrays from Part 1 (change maps, image logits), so no model has to run for the review sheets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from second_look import metrics as M
from second_look.dataset import CACHE
from second_look.eval import val_threshold

METHOD = "deltadino_seed0"
HELDOUT_TASKS = (6, 8)


def labels_v2_rows(split: str) -> pd.DataFrame:
    """label_v2 / gray_v2 / max_rot_v2 aligned with the split's cache meta (row i = cache row i). Splits built after Part 1.5 (probe_static, test_policy2)
    carry v2 labels directly in meta.parquet and have no labels_v2.parquet."""
    from second_look.scene import ROOT
    full = pd.read_parquet(CACHE / split / "meta.parquet")
    f = ROOT / "data/pairs" / split / "labels_v2.parquet"
    if not f.exists():
        return pd.DataFrame({"label_v2": full.label.values, "gray_v2": full.gray.values.astype(bool), "max_rot_v2": full.max_rot_deg.values})
    uid = (full.pair_id + "#" + full.groupby("pair_id").cumcount().astype(str)).values
    return pd.read_parquet(f).set_index("uid").loc[uid].reset_index(drop=True)


def val_threshold_v2(method: str) -> float:
    """F1-max threshold on val with label_v2 (non-gray v2)."""
    meta, sc = M.load_scores(method, "val")
    l = labels_v2_rows("val").iloc[sc["row"]]
    ng = ~l.gray_v2.to_numpy(bool)
    return M.best_f1_threshold(l.label_v2.to_numpy()[ng], sc["score"][ng])


class Split:
    """Scores + metadata (aligned) + lazy memmaps for one split of one method. ``labels="v2"`` replaces label / gray / max_rot_deg by
    label_v2 / gray_v2 / max_rot_v2 and (unless ``thr`` is given) uses the val threshold derived with v2 labels."""

    def __init__(self, split: str, method: str = METHOD, thr: float | None = None, labels: str = "v1"):
        self.name, self.method, self.labels = split, method, labels
        self.meta, self.sc = M.load_scores(method, split)          # meta aligned with sc arrays
        if labels == "v2":
            l = labels_v2_rows(split).iloc[self.sc["row"]]
            self.meta = self.meta.copy()
            self.meta["label"], self.meta["gray"], self.meta["max_rot_deg"] = l.label_v2.values, l.gray_v2.values.astype(bool), l.max_rot_v2.values
            self.thr = val_threshold_v2(method) if thr is None else thr
        else:
            self.thr = val_threshold(method) if thr is None else thr
        self.score = self.sc["score"]
        self.pred = self.score >= self.thr
        self.arm, self.chg, self.pmap = self.sc["arm"], self.sc["chg"], self.sc["pmap"]
        self._mm = None

    def _open(self):
        if self._mm is None:
            d = CACHE / self.name
            self._mm = tuple(np.load(d / f, mmap_mode="r") for f in ("ref.npy", "set.npy", "masks.npy"))
        return self._mm

    def images(self, i: int):
        """(ref, set uint8 224x224x3, masks bool 4x224x224: arm_ref, arm_set, chg_ref, chg_set) for score index i."""
        ref, st, msk = self._open()
        r = int(self.sc["row"][i])
        m = np.unpackbits(np.asarray(msk[r]), axis=1).reshape(4, 224, 224).astype(bool)
        return np.array(ref[r]), np.array(st[r]), m

    def argmax_patch(self, i: int):
        z = np.where(self.arm[i], -np.inf, self.pmap[i])
        if not np.isfinite(z).any():
            return None
        return np.unravel_index(int(np.argmax(z)), z.shape)

    def pointing_hit(self, i: int):
        """True/False if defined (argmax non-arm patch on a changed patch), None if no changed patch / no non-arm patch."""
        a = self.argmax_patch(i)
        if a is None or not self.chg[i].any():
            return None
        return bool(self.chg[i][a])

    @property
    def nongray(self) -> np.ndarray:
        return ~self.meta["gray"].to_numpy(bool)

    @property
    def y(self) -> np.ndarray:
        return self.meta["label"].to_numpy()


def stratified(idx: np.ndarray, groups: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Pick ``n`` indices spread as evenly as possible over the distinct values of ``groups[idx]``."""
    idx = np.asarray(idx)
    if len(idx) <= n:
        return [int(i) for i in idx]
    by = {g: list(rng.permutation(idx[groups[idx] == g])) for g in np.unique(groups[idx])}
    out = []
    while len(out) < n and any(by.values()):
        for g in sorted(by):
            if by[g] and len(out) < n:
                out.append(int(by[g].pop()))
    return out


def select_sheets(seed: int = 0) -> dict[str, tuple[str, list[int]]]:
    """name -> (split label, list of score indices) per SPEC §4. Multi-split sheets (S-D) return per-row splits."""
    rng = np.random.default_rng(seed)
    val, test, held, pol = (Split(s) for s in ("val", "test", "test_heldout_tasks", "test_policy"))
    out = {}

    # S-A: 16 S1, 16 G1-G5, 16 H1-H4 from val
    g = val.meta["gen"].to_numpy()
    ng = val.nongray
    s1 = np.flatnonzero(ng & (g == "S1"))
    gg = np.flatnonzero(ng & np.isin(g, ["G1", "G2", "G3", "G4", "G5"]))
    hh = np.flatnonzero(ng & np.isin(g, ["H1", "H2", "H3", "H4"]))
    out["S-A_val_random"] = [("val", i) for i in stratified(s1, g, 16, rng) + stratified(gg, g, 16, rng) + stratified(hh, g, 16, rng)]

    # S-B: test positives correctly called changed but argmax not on a changed patch (stratified by generator)
    hit = np.array([test.pointing_hit(i) for i in range(len(test.meta))], dtype=object)
    cand = np.flatnonzero(test.nongray & (test.y == 1) & test.pred & (hit == False))   # noqa: E712
    out["S-B_pointing_misses"] = [("test", i) for i in stratified(cand, test.meta["gen"].to_numpy(), 30, rng)]

    # S-C: policy unchanged pairs predicted changed
    cand = np.flatnonzero(pol.nongray & (pol.y == 0) & pol.pred)
    out["S-C_policy_false_alarms"] = [("test_policy", i) for i in stratified(cand, pol.meta["task"].to_numpy(), 30, rng)]

    # S-D: test + policy positives with rotation 5-15 deg and translation < 1 cm that were missed
    rows = []
    for sp in (test, pol):
        m = sp.meta
        c = np.flatnonzero(sp.nongray & (sp.y == 1) & ~sp.pred & (m["max_rot_deg"].to_numpy() >= 5) & (m["max_rot_deg"].to_numpy() <= 15) & (m["max_trans_cm"].to_numpy() < 1))
        rows += [(sp.name, int(i)) for i in c]
    rng.shuffle(rows)
    out["S-D_rotation_misses"] = rows[:30]

    # S-E: held-out positives with big change that were missed
    m = held.meta
    c = np.flatnonzero(held.nongray & (held.y == 1) & ~held.pred & ((m["max_trans_cm"].to_numpy() > 2) | (m["max_rot_deg"].to_numpy() > 15)))
    out["S-E_heldout_misses"] = [("test_heldout_tasks", i) for i in stratified(c, m["task"].to_numpy(), 30, rng)]
    out["_pool_sizes"] = {k: None for k in out}
    return out


# ---------------------------------------------------------------------------------------------------
# C2: pointing metrics
# ---------------------------------------------------------------------------------------------------
def cheb_dist_to_set(idx: tuple[int, int], mask: np.ndarray) -> int | None:
    """Chebyshev distance (patches) from ``idx`` to the nearest True cell of a 16x16 mask; None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(np.max(np.stack([np.abs(ys - idx[0]), np.abs(xs - idx[1])]), axis=0).min())


def pointing_row(sp: "Split", i: int, changed_pixels_under_arm: float | None = None) -> dict:
    """All C2-A metrics for one pair (None where undefined)."""
    arm, chg, z = sp.arm[i], sp.chg[i], sp.pmap[i]
    vis = chg & ~arm
    a = sp.argmax_patch(i)
    out = {"n_changed_visible": int(vis.sum()), "frac_changed_under_arm": changed_pixels_under_arm}
    if a is None or not chg.any():
        out.update(P_strict=None, P_adj=None, P_top3=None, P_visible=None, mass_on_target=None, dist_argmax=None, argmax_near_arm=None)
        return out
    d = cheb_dist_to_set(a, chg)
    zz = np.where(arm, -np.inf, z)
    top3 = np.argsort(zz.ravel())[::-1][:3]
    top3 = [np.unravel_index(int(t), zz.shape) for t in top3 if np.isfinite(zz.ravel()[t])]
    prob = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    tot = float(prob[~arm].sum())
    strict = bool(chg[a])
    out.update(
        P_strict=strict, P_adj=bool(d <= 1), P_top3=bool(any(chg[t] for t in top3)),
        P_visible=(strict if vis.sum() >= 2 else None),
        mass_on_target=(float(prob[vis].sum() / tot) if tot > 0 else None),
        dist_argmax=int(d), argmax_near_arm=bool(cheb_dist_to_set(a, arm) is not None and cheb_dist_to_set(a, arm) <= 1))
    return out


def changed_under_arm_fraction(sp: "Split", i: int) -> float | None:
    """Fraction of changed-object pixels (ref U set) that lie inside the model's (dilated) arm patch mask."""
    _, _, m = sp.images(i)
    chg_px = m[2] | m[3]
    n = int(chg_px.sum())
    if n == 0:
        return None
    armp = np.kron(sp.arm[i].astype(np.uint8), np.ones((14, 14), np.uint8)).astype(bool)
    return float((chg_px & armp).sum() / n)


# ---------------------------------------------------------------------------------------------------
# C2-B / C2-C: image edits (torch, GPU or CPU). All inputs float images (B,3,H,W) in [0,1], masks bool (B,H,W)
# ---------------------------------------------------------------------------------------------------
def dilate_px(mask, n: int):
    import torch
    import torch.nn.functional as F
    if n <= 0:
        return mask
    return F.max_pool2d(mask.float().unsqueeze(1), 2 * n + 1, stride=1, padding=n).squeeze(1) > 0.5


def remove_change(ref, st, chg_ref, chg_set, arm_ref, arm_set, dil: int = 3):
    """I_set' = I_set with every changed-object pixel (ref U set masks, dilated ``dil`` px, outside the arm masks)
    replaced by the I_ref pixel. If the change mask is empty the frame is returned unchanged."""
    import torch
    m = dilate_px(chg_ref | chg_set, dil) & ~(arm_ref | arm_set)
    return torch.where(m.unsqueeze(1), ref, st), m


def swap_arm(ref, st, arm_ref, arm_set):
    """I_set' = I_set with its arm pixels replaced by I_ref content wherever the reference arm was not (background
    that was visible in the reference). Where both arms overlap the arm stays."""
    import torch
    m = arm_set & ~arm_ref
    return torch.where(m.unsqueeze(1), ref, st), m


@__import__("torch").no_grad()
def logits_for(model, ref, st, arm_patch):
    """Image logits (B,) for float images and a (B,16,16) bool arm patch mask."""
    import torch
    with torch.autocast(ref.device.type, dtype=torch.bfloat16):
        s, _ = model(ref, st, model.apply_static(arm_patch))
    return s.float()
