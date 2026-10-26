"""Gate threshold recalibration for the in-loop setting (deviations.md 2026-10-10 "gate threshold recalibrated"; rule written before this ran).

  python scripts/p2_gate_cal.py [--probe results/part2/p2_3b] [--boot 2000] [--write]

Data: the P2.3b CAL pairs (alarm-time frames: reference at the target, settled after the 10-step pause), SEEN tasks, all alarms, gray pairs excluded, per target.
Ops: `fpr05` / `fpr03` = the lowest threshold with in-loop FPR <= 5% / 3% on the unchanged pairs: with n unchanged pairs and k = floor(alpha n), t = nextafter(s_(k+1), +inf), s_(k+1) the
(k+1)-th largest unchanged score, so that the loop's rule (`s >= t` -> CHANGED) flags at most k unchanged pairs. One threshold per target for all tasks. Seeds 0, 1, 2: seed 0 uses the stored
scores (checked against a rescoring of the stored frames), seeds 1 and 2 score the same stored frames. CIs: cluster bootstrap over episodes. Cross-fit: 5 folds by init state.
--write: configs/part2_gate_cal.yaml and results/part2/notes/p2_gate_cal.md (results/part2/p2_gate_cal.json always).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HELD = (6, 8)
TARGETS = ("L1", "L2", "L4", "LC4", "L6", "L10")
OPS = {"fpr05": 0.05, "fpr03": 0.03}
FOLDS = [(40, 41), (42, 43), (44, 45), (46, 47), (48, 49)]


def load_pairs(probe: Path) -> list[dict]:
    rows = []
    for f in sorted(probe.glob("ep_*.json.gz")):
        r = json.load(gzip.open(f))
        for a in r["alarms"]:
            for tn, t in a["targets"].items():
                rows.append({"ep": (r["task"], r["init"], r["seed_id"]), "task": r["task"], "init": r["init"], "target": tn, "label": t["label"], "gray": t["gray"], "s0": t["s"],
                             "ref": t["ref"], "set": t["set"], "alarm": a["alarm_idx"]})
    return rows


def score_all(rows: list[dict], probe: Path, seeds=(0, 1, 2)) -> dict:
    """Score every pair with every seed on the stored frames. Returns {seed: array aligned with rows}. Seed 0 is checked against the stored scores."""
    from second_look.gate import DeltaGate
    gate = DeltaGate()
    out = {s: np.zeros(len(rows)) for s in seeds}
    by_ep = defaultdict(list)
    for i, r in enumerate(rows):
        by_ep[r["ep"]].append(i)
    for ep, idx in by_ep.items():
        z = np.load(probe / f"ep_{ep[0]}_{ep[1]}_{ep[2]}.npz")
        for i in idx:
            r = rows[i]
            for s in seeds:
                out[s][i] = gate.score(z[r["ref"]], z[r["set"]], s)[0]
    return out


def lowest_threshold(unch: np.ndarray, alpha: float) -> float:
    n = len(unch)
    k = int(np.floor(alpha * n + 1e-9))
    s = np.sort(unch)[::-1]
    if k >= n:
        return float(np.min(unch) - 1.0)
    return float(np.nextafter(s[k], np.inf))                   # s[k] = the (k+1)-th largest


def rates(sc: np.ndarray, lab: np.ndarray, t: float) -> tuple[float | None, float | None]:
    unch, chg = sc[lab == 0], sc[lab == 1]
    return (float((unch >= t).mean()) if len(unch) else None, float((chg >= t).mean()) if len(chg) else None)


def analyse(rows, scores: np.ndarray, boot: int, alpha: float, rng) -> dict:
    sc = scores
    lab = np.array([r["label"] for r in rows])
    task = np.array([r["task"] for r in rows])
    init = np.array([r["init"] for r in rows])
    epi = np.array([hash(r["ep"]) for r in rows])
    seen = ~np.isin(task, HELD)
    t = lowest_threshold(sc[seen & (lab == 0)], alpha)
    fpr, rec = rates(sc[seen], lab[seen], t)
    res = {"threshold": t, "n_unchanged": int((seen & (lab == 0)).sum()), "n_changed": int((seen & (lab == 1)).sum()), "fpr": fpr, "recall": rec}
    # cluster bootstrap over episodes (seen)
    eps = sorted({r["ep"] for r, s in zip(rows, seen) if s})
    ep_idx = {e: np.array([i for i, r in enumerate(rows) if r["ep"] == e and seen[i]]) for e in eps}
    tb, fb, rb = [], [], []
    for _ in range(boot):
        pick = rng.integers(0, len(eps), len(eps))
        idx = np.concatenate([ep_idx[eps[j]] for j in pick])
        if not ((lab[idx] == 0).any() and (lab[idx] == 1).any()):
            continue
        tt = lowest_threshold(sc[idx][lab[idx] == 0], alpha)
        tb.append(tt)
        f_, r_ = rates(sc[idx], lab[idx], t)                      # FPR / recall at the FIXED fitted threshold
        fb.append(f_)
        rb.append(r_)
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res.update(threshold_ci=pc(tb), fpr_ci=pc(fb), recall_ci=pc(rb))
    # per task at the fitted threshold
    res["per_task"] = {}
    for k in range(10):
        m = task == k
        f_, r_ = rates(sc[m], lab[m], t)
        res["per_task"][str(k)] = {"n_unchanged": int((m & (lab == 0)).sum()), "n_changed": int((m & (lab == 1)).sum()), "fpr": f_, "recall": r_}
    # held-out tasks at the fitted threshold (descriptive)
    h = ~seen
    f_, r_ = rates(sc[h], lab[h], t)
    res["held_out"] = {"n_unchanged": int((h & (lab == 0)).sum()), "n_changed": int((h & (lab == 1)).sum()), "fpr": f_, "recall": r_}
    # 5-fold cross-fit by init state
    folds = []
    for fo in FOLDS:
        te = seen & np.isin(init, fo)
        tr = seen & ~np.isin(init, fo)
        tf = lowest_threshold(sc[tr & (lab == 0)], alpha)
        f_, r_ = rates(sc[te], lab[te], tf)
        folds.append({"inits": list(fo), "threshold": tf, "n_unchanged": int((te & (lab == 0)).sum()), "n_changed": int((te & (lab == 1)).sum()), "fpr": f_, "recall": r_,
                      "fp": int(((sc >= tf) & te & (lab == 0)).sum()), "tp": int(((sc >= tf) & te & (lab == 1)).sum())})
    ok = [f for f in folds if f["fpr"] is not None and f["recall"] is not None]
    pooled_fp = sum(f["fp"] for f in folds)
    pooled_tp = sum(f["tp"] for f in folds)
    nu = sum(f["n_unchanged"] for f in folds)
    nc = sum(f["n_changed"] for f in folds)
    res["cross_fit"] = {"folds": folds, "fpr_mean": float(np.mean([f["fpr"] for f in ok])), "fpr_worst": float(max(f["fpr"] for f in ok)),
                        "recall_mean": float(np.mean([f["recall"] for f in ok])), "recall_worst": float(min(f["recall"] for f in ok)),
                        "fpr_pooled": pooled_fp / nu if nu else None, "recall_pooled": pooled_tp / nc if nc else None}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=str(ROOT / "results" / "part2" / "p2_3b"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    probe = Path(a.probe)
    rows = load_pairs(probe)
    cache = ROOT / "results" / "part2" / "p2_3b_scores.npz"
    if cache.exists() and len(np.load(cache)["s0"]) == len(rows):
        sc = {s: np.load(cache)[f"s{s}"] for s in (0, 1, 2)}
    else:
        sc = score_all(rows, probe)
        np.savez(cache, **{f"s{s}": v for s, v in sc.items()})
    s0 = np.array([r["s0"] for r in rows])
    same = float(np.abs(sc[0] - s0).max())
    print(f"{len(rows)} pairs; rescored seed 0 vs stored: max |diff| = {same:.3g}")
    keep = np.array([not r["gray"] for r in rows])
    rows_ng = [r for r, k in zip(rows, keep) if k]
    rng = np.random.default_rng(0)
    out = {"n_pairs_total": len(rows), "n_gray_dropped": int((~keep).sum()), "seed0_rescore_max_abs_diff": same, "stats": {}, "thresholds": {}}
    for seed in (0, 1, 2):
        out["stats"][seed], out["thresholds"][seed] = {}, {}
        for op, alpha in OPS.items():
            out["stats"][seed][op], out["thresholds"][seed][op] = {}, {}
            for tg in TARGETS:
                m = np.array([r["target"] == tg for r in rows_ng])
                rr = [r for r, k in zip(rows_ng, m) if k]
                res = analyse(rr, sc[seed][keep][m], a.boot, alpha, rng)
                out["stats"][seed][op][tg] = res
                out["thresholds"][seed][op][tg] = res["threshold"]
    (ROOT / "results" / "part2").mkdir(exist_ok=True)
    (ROOT / "results" / "part2" / "p2_gate_cal.json").write_text(json.dumps(out, indent=1, default=float))
    fp = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    L = ["# Gate threshold recalibration (in-loop, CAL P2.3b pairs, seen tasks, all alarms, gray excluded)", "",
         "Rule and data: `results/part2/notes/deviations.md` (2026-10-10 'gate threshold recalibrated'). `fpr05` / `fpr03` = lowest threshold with in-loop FPR <= 5% / 3%; one per target; the loop flags CHANGED when s >= threshold. "
         f"Original thresholds (all targets): F1-max {1.7639:+.3f}, recall>=0.99 {-2.1035:+.3f} (seed 0). Seed-0 rescoring of the stored frames reproduces the stored scores (max |diff| {same:.2g}).", ""]
    for seed in (0, 1, 2):
        L += [f"## Seed {seed}", "", "| op | target | n unch/chg | threshold [95% CI] | FPR [95% CI] | recall [95% CI] | cross-fit FPR mean / worst | cross-fit recall mean / worst | held-out FPR / recall | task 7 FPR / recall (n) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for op in OPS:
            for tg in TARGETS:
                s = out["stats"][seed][op][tg]
                cf = s["cross_fit"]
                t7 = s["per_task"]["7"]
                L.append(f"| {op} | {tg} | {s['n_unchanged']}/{s['n_changed']} | {s['threshold']:+.2f} [{s['threshold_ci'][0]:+.2f}, {s['threshold_ci'][1]:+.2f}] | {fp(s['fpr'])} [{fp(s['fpr_ci'][0])}, {fp(s['fpr_ci'][1])}] | "
                         f"{fp(s['recall'])} [{fp(s['recall_ci'][0])}, {fp(s['recall_ci'][1])}] | {fp(cf['fpr_mean'])} / {fp(cf['fpr_worst'])} | {fp(cf['recall_mean'])} / {fp(cf['recall_worst'])} | "
                         f"{fp(s['held_out']['fpr'])} / {fp(s['held_out']['recall'])} | {fp(t7['fpr'])} / {fp(t7['recall'])} ({t7['n_unchanged']}/{t7['n_changed']}) |")
        L.append("")
    L += ["## Per task at the fitted threshold (seed 0; FPR / recall; n unchanged / changed)", "", "| op | target | " + " | ".join(str(k) for k in range(10)) + " |", "|---|---|" + "---|" * 10]
    for op in OPS:
        for tg in TARGETS:
            pt = out["stats"][0][op][tg]["per_task"]
            L.append(f"| {op} | {tg} | " + " | ".join(f"{fp(pt[str(k)]['fpr'])} / {fp(pt[str(k)]['recall'])} ({pt[str(k)]['n_unchanged']}/{pt[str(k)]['n_changed']})" for k in range(10)) + " |")
    txt = "\n".join(L) + "\n"
    print(txt[:6000])
    if a.write:
        (ROOT / "results" / "part2" / "notes" / "p2_gate_cal.md").write_text(txt)
        cfg = {"description": "Gate threshold recalibration for the in-loop setting (P2.3b CAL pairs, seen tasks, all alarms, gray excluded). Rule: results/part2/notes/deviations.md 2026-10-10. "
                              "The loop calls an alarm CHANGED when the DeltaDINO image logit s >= threshold. fpr05 / fpr03: lowest threshold with in-loop FPR <= 5% / 3%, one per target for all tasks.",
               "source": "results/part2/p2_3b (scripts/p2_probe.py), scripts/p2_gate_cal.py", "n_pairs": len(rows), "n_gray_dropped": int((~keep).sum()),
               "thresholds": {int(s): {op: {tg: float(v) for tg, v in d.items()} for op, d in o.items()} for s, o in out["thresholds"].items()},
               "cal_stats": {int(s): {op: {tg: {k: v for k, v in x.items() if k not in ("per_task", "cross_fit")} | {"cross_fit_fpr_mean": x["cross_fit"]["fpr_mean"], "cross_fit_fpr_worst": x["cross_fit"]["fpr_worst"],
                                                                                  "cross_fit_recall_mean": x["cross_fit"]["recall_mean"], "cross_fit_recall_worst": x["cross_fit"]["recall_worst"],
                                                                                  "task7": x["per_task"]["7"]} for tg, x in d.items()} for op, d in o.items()} for s, o in out["stats"].items()}}
        def plain(o):
            if isinstance(o, dict):
                return {(k if not isinstance(k, np.generic) else k.item()): plain(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [plain(v) for v in o]
            if isinstance(o, np.generic):
                return o.item()
            return o
        yaml.safe_dump(plain(cfg), open(ROOT / "configs" / "part2_gate_cal.yaml", "w"), sort_keys=False)
        print("written configs/part2_gate_cal.yaml and results/part2/notes/p2_gate_cal.md")


if __name__ == "__main__":
    main()
