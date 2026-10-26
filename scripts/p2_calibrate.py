"""P2.3 calibration from the CAL run (P0 only, STAC on): fits the accel per-coordinate scale, the CUSUM parameters of both detectors, the P5 hazard, reports CAL behaviour and
writes the frozen config.   python scripts/p2_calibrate.py [--cal results/part2/cal] [--write-config]

Fit (SPEC_part2 §6 P2.3): scale = std of the flattened executed window (10 steps x 7 dims) over ALL CAL decisions and denoise iterates; per detector, from CAL SUCCESSES only:
mu0, sigma pooled over decisions, k = 0.25 sigma, h = the ceil((M+1)(1-alpha))-th smallest per-episode CUSUM peak, alpha = 0.1 (M = number of success episodes).
P5 hazard: empirical first-alarm hazard of accel by decision index over ALL CAL episodes, numerator and denominator smoothed over +-1 decision.
Outputs: results/part2/cal_report.json, results/part2/cal_first_alarm.png, results/part2/notes/p2_3_cal.md (local), configs/part2_frozen.yaml (with --write-config).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from second_look import detectors as D
from second_look.cusum import K_FACTOR, CusumParams, conformal_height, first_alarm, peaks
from second_look.scene import ROOT

ALPHA = 0.1
LIMIT = 300
NDEC = 30


def load(cal: Path) -> list[dict]:
    eps = []
    for f in sorted(glob.glob(str(cal / "ep_*.json.gz"))):
        r = json.load(gzip.open(f))
        a = r["arms"]["P0"]
        z = np.load(f.replace(".json.gz", ".npz"))["windows"]                      # (D, 11, 10, 7)
        log = a["decisions_log"]
        assert len(log) == len(z), f
        eps.append({"task": r["task"], "init": r["init"], "seed": r["seed_id"], "success": bool(a["success"]), "n_dec": len(log), "steps": int(a["executed"]),
                    "stac": np.array([d["stac"] for d in log], np.float64), "windows": z, "secs": r["secs"], "git": r["git"]})
    return eps


def fit(success_series: list[np.ndarray], alpha=ALPHA) -> CusumParams:
    pooled = np.concatenate(success_series)
    mu0, sigma = float(pooled.mean()), float(pooled.std())
    return CusumParams.make(mu0, sigma, conformal_height(peaks(success_series, mu0, K_FACTOR * sigma), alpha))


def crossfit_fpr(eps: list[dict], key: str, folds: int = 5) -> dict:
    """Fit on the successes of the other folds (folds by LIBERO init state, both noise seeds of an init together), count held-out successes with an alarm."""
    inits = sorted({e["init"] for e in eps})
    fold = {i: k % folds for k, i in enumerate(inits)}
    n_alarm = n = 0
    for f in range(folds):
        tr = [e[key] for e in eps if e["success"] and fold[e["init"]] != f]
        te = [e[key] for e in eps if e["success"] and fold[e["init"]] == f]
        p = fit(tr)
        n_alarm += sum(first_alarm(x, p) is not None for x in te)
        n += len(te)
    return {"folds": folds, "held_out_successes": n, "false_alarm_rate": n_alarm / max(n, 1)}


def hazard(first: list[int | None], lengths: list[int], ndec=NDEC) -> np.ndarray:
    """P(first alarm at decision d | still running and not yet alarmed at d), counts smoothed over +-1 decision."""
    num = np.zeros(ndec); den = np.zeros(ndec)
    for f, n in zip(first, lengths):
        last = (f if f is not None else n - 1)
        for d in range(min(last, ndec - 1) + 1):
            den[d] += 1
        if f is not None and f < ndec:
            num[f] += 1
    pad = lambda v: np.concatenate([[v[0]], v, [v[-1]]])
    sm = lambda v: np.convolve(pad(v), np.ones(3), mode="valid")
    return np.where(sm(den) > 0, sm(num) / np.maximum(sm(den), 1e-9), 0.0)


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="results/part2/cal")
    ap.add_argument("--write-config", action="store_true")
    a = ap.parse_args()
    eps = load(ROOT / a.cal)
    n_ep = len(eps)
    # ---- accel
    scale = D.fit_accel_scale([D.exec_window(w[i]) for e in eps for w in [e["windows"]] for i in range(len(w))])
    for e in eps:
        e["accel"] = np.array([D.accel(w, scale) for w in e["windows"]], np.float64)
    succ = [e for e in eps if e["success"]]
    M = len(succ)
    P = {k: fit([e[k] for e in succ]) for k in ("accel", "stac")}
    rep = {"episodes": n_ep, "success_episodes": M, "alpha": ALPHA, "cusum_rank": int(np.ceil((M + 1) * (1 - ALPHA))), "params": {k: vars(v) for k, v in P.items()}, "secs_per_episode_mean": float(np.mean([e["secs"] for e in eps]))}
    # ---- success rates
    per_task = {}
    for t in range(10):
        s = [e["success"] for e in eps if e["task"] == t]
        per_task[t] = {"n": len(s), "success": float(np.mean(s)) if s else None, "median_success_steps": float(np.median([e["steps"] for e in eps if e["task"] == t and e["success"]])) if any(e["success"] for e in eps if e["task"] == t) else None}
    rep["success"] = {"overall": float(np.mean([e["success"] for e in eps])), "per_task": per_task, "part1_overall_65_5_note": "Part 1: 200 episodes, init states 0-19, one noise draw each, overall 0.655",
                      "seen_tasks": float(np.mean([e["success"] for e in eps if e["task"] not in (6, 8)])), "heldout_tasks_6_8": float(np.mean([e["success"] for e in eps if e["task"] in (6, 8)]))}
    # ---- alarms
    rep["detectors"] = {}
    first_acc = None
    for k in ("accel", "stac"):
        p = P[k]
        fa = [first_alarm(e[k], p) for e in eps]
        if k == "accel":
            first_acc = fa
        fs = [f for f, e in zip(fa, eps) if e["success"]]
        ff = [(f, e) for f, e in zip(fa, eps) if not e["success"]]
        n_fail = len(ff)
        alarmed = [(f, e) for f, e in ff if f is not None]
        dec = np.array([f for f, _ in alarmed], float)
        steps = dec * 10                                  # decision d starts at step 10 d: the alarm is raised before its chunk is executed
        left = LIMIT - steps
        rep["detectors"][k] = {
            "realized_fpr_in_sample": float(np.mean([f is not None for f in fs])), "crossfit": crossfit_fpr(eps, k),
            "n_failed": n_fail, "failed_alarmed": len(alarmed), "failed_alarmed_frac": len(alarmed) / max(n_fail, 1),
            "first_alarm_decision": {"median": q(dec, 50), "q25": q(dec, 25), "q75": q(dec, 75), "min": float(dec.min()) if len(dec) else None, "max": float(dec.max()) if len(dec) else None},
            "first_alarm_step": {"median": q(steps, 50), "q25": q(steps, 25), "q75": q(steps, 75)},
            "budget_left_at_alarm_steps": {"median": q(left, 50), "q25": q(left, 25), "q75": q(left, 75)},
            "alarms_leaving_at_least_steps": {str(x): float(np.mean(left >= x)) if len(left) else None for x in (30, 60, 100, 150, 200)},
            "success_false_alarm_first_decision": {"median": q([f for f in fs if f is not None], 50), "n": int(sum(f is not None for f in fs))},
            "by_task_failed_alarmed": {str(t): [int(sum(1 for f, e in alarmed if e["task"] == t)), int(sum(1 for f, e in ff if e["task"] == t))] for t in range(10)},
        }
    # per-episode AUROC of the max CUSUM (indicative; compare with the old project: accel 0.967, STAC 0.912)
    from second_look import metrics as M_
    y = np.array([0 if e["success"] else 1 for e in eps])
    for k in ("accel", "stac"):
        pk = peaks([e[k] for e in eps], P[k].mu0, P[k].k)
        rep["detectors"][k]["auroc_max_cusum_episode"] = float(M_.auroc(y, pk))
    # ---- P5 hazard
    lengths = [e["n_dec"] for e in eps]
    h5 = hazard(first_acc, lengths)
    rep["p5_hazard"] = [float(x) for x in h5]
    rep["p5_hazard_raw_first_alarm_counts"] = np.bincount([f for f in first_acc if f is not None], minlength=NDEC).tolist()
    # ---- git / sanity
    rep["decisions_checked_by_image_sanity"] = int(sum(e["n_dec"] for e in eps))
    rep["code_git_of_cal_run"] = sorted({e["git"] for e in eps})
    rep["accel_scale"] = [float(x) for x in scale]
    out = ROOT / "results" / "part2" / "cal_report.json"
    out.write_text(json.dumps(rep, indent=1))
    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for k, c in (("accel", "tab:blue"), ("stac", "tab:orange")):
        p = P[k]
        d = [first_alarm(e[k], p) for e in eps if not e["success"]]
        ax[0].hist([x for x in d if x is not None], bins=np.arange(-0.5, 31.5, 1), alpha=0.6, color=c, label=f"{k}: failed episodes ({sum(x is not None for x in d)}/{len(d)} alarmed)")
    ax[0].set_xlabel("decision index of the first alarm (x10 = step)"); ax[0].legend(); ax[0].set_title("first alarm on failures")
    ax[1].bar(range(NDEC), h5); ax[1].set_title("P5 hazard (accel, smoothed)"); ax[1].set_xlabel("decision index")
    ax[2].bar(range(10), [per_task[t]["success"] or 0 for t in range(10)]); ax[2].axhline(0.655, color="k", ls="--", lw=1); ax[2].set_title("CAL success per task (dashed: Part 1 overall 0.655)"); ax[2].set_xlabel("task")
    fig.tight_layout(); fig.savefig(ROOT / "results/part2/cal_first_alarm.png", dpi=110)
    print(json.dumps({k: rep[k] for k in ("episodes", "success_episodes", "params")}, indent=1))
    print("success", rep["success"]["overall"], {t: (round(v["success"], 2) if v["success"] is not None else None) for t, v in per_task.items()})
    for k in ("accel", "stac"):
        d = rep["detectors"][k]
        print(k, "FPR in-sample", round(d["realized_fpr_in_sample"], 3), "crossfit", round(d["crossfit"]["false_alarm_rate"], 3), "failed alarmed", d["failed_alarmed"], "/", d["n_failed"],
              "first alarm decision median/IQR", d["first_alarm_decision"]["median"], d["first_alarm_decision"]["q25"], d["first_alarm_decision"]["q75"], "AUROC", round(d["auroc_max_cusum_episode"], 3))
    print("P5 hazard", np.round(h5, 3).tolist())
    if a.write_config:
        write_config(rep, scale)


def write_config(rep: dict, scale: np.ndarray) -> None:
    import yaml
    ops = json.load(open(ROOT / "results/part2/c_operating_points.json"))
    thr = {}
    ck = {}
    for s in (0, 1, 2):
        m = f"deltadino_v2static_seed{s}"
        thr[s] = {k: float(v) for k, v in ops[m]["thresholds"].items()}
        p = ROOT / f"results/part1/deltadino_v2_static/seed{s}/best.pt"
        ck[s] = {"path": f"results/part1/deltadino_v2_static/seed{s}/best.pt", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    pa, ps = rep["params"]["accel"], rep["params"]["stac"]
    cfg = {
        "frozen_by": "scripts/p2_calibrate.py --write-config", "frozen_from": "results/part2/cal (CAL run: 10 tasks x init 40-49 x seeds p2cal-1, p2cal-2; P0, STAC on)",
        "cal_episodes": rep["episodes"], "cal_success_episodes": rep["success_episodes"], "calibrate_script_git_parent": git, "cal_run_code_git": rep["code_git_of_cal_run"],
        "alpha": ALPHA, "k_factor": K_FACTOR, "budget_limit": LIMIT, "pause_steps": 10, "n_exec": 10, "ring": 12, "stac_k": 8,
        "accel": {"mu0": pa["mu0"], "sigma": pa["sigma"], "h": pa["h"], "k": pa["k"]}, "accel_scale": [float(x) for x in scale],
        "stac": {"mu0": ps["mu0"], "sigma": ps["sigma"], "h": ps["h"], "k": ps["k"]},
        "p5_hazard": rep["p5_hazard"],
        "deltadino": {"model": "C (static mask only, deltadino_v2_static)", "main_seed": 0, "checkpoints": ck,
                      "thresholds_val_label_v2": thr, "thresholds_source": "results/part2/notes/c_operating_points.md (val only; results/part2/c_operating_points.json)"},
        "episode_sets": {"cal": "tasks 0-9, init 40-49, p2cal-1/2 (200)", "dev": "tasks 0-9, init 0-19, p2dev-1 (200)", "test": "tasks 0-9, init 20-49, p2test-1/2/3 (900)"},
        "note": "Arms (DEV grid, TEST arms) are NOT in this file: they live in configs/part2_dev_grid.yaml (P2.5) and configs/part2_test_arms.yaml (P2.6), so this file stays frozen.",
    }
    path = ROOT / "configs" / "part2_frozen.yaml"
    path.write_text("# Part 2 frozen calibration (P2.3). Do not edit; a change means CAL or the DeltaDINO thresholds changed and must be a new, dated, committed file.\n" + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=140))
    print("wrote", path)


if __name__ == "__main__":
    main()
