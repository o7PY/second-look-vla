"""The Part 2 runner: CAL / DEV / TEST (and a throw-away `smoke` set), resumable, one process = one worker with ONE policy env and ONE render Scene.

  scripts/safe_run.sh python scripts/p2_run.py --set smoke --config <yaml> --limit 2          # pipeline check, its own folder, never feeds a result
  scripts/safe_run.sh python scripts/p2_run.py --set cal  [--worker 0 --workers 2]            # NOT started until the user says so
  scripts/safe_run.sh python scripts/p2_run.py --set dev  --config configs/part2_frozen.yaml --gate-cal configs/part2_gate_cal.yaml --arms configs/part2_dev_grid.yaml
  scripts/safe_run.sh python scripts/p2_run.py --set test --config configs/part2_frozen.yaml --gate-cal configs/part2_gate_cal.yaml --arms configs/part2_test_arms.yaml  # refused unless sl.p2guard.check_test() passes

Episode sets (SPEC_part2 §5): cal = 10 tasks x init 40-49 x seeds p2cal-1,2 (200); dev = 10 tasks x init 0-19 x p2dev-1 (200); test = 10 tasks x init 20-49 x
p2test-1..3 (900). The split string in the noise key is the set name. Per episode one file `<out>/ep_<task>_<init>_<seed>.json.gz` (+ `.npz` with the gate pairs),
written atomically (tmp + rename); finished episodes are skipped on restart. Never more than 2 workers (`--worker i --workers n`, n <= 2).

CAL runs P0 only with STAC on and stores the executed-window denoise paths (the accel scale and the CUSUM parameters are fitted from them offline, scripts/p2_calibrate.py).
The frozen config (yaml) holds: accel {mu0, sigma, h}, accel_scale [70], stac {mu0, sigma, h}, p5_hazard [..], deltadino.thresholds_val_label_v2 {seed: {op: logit}}; the arms come from --arms.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from second_look.scene import ROOT  # noqa: E402

SETS = {
    "cal": {"inits": range(40, 50), "seeds": ["p2cal-1", "p2cal-2"]},
    "dev": {"inits": range(0, 20), "seeds": ["p2dev-1"]},
    "test": {"inits": range(20, 50), "seeds": ["p2test-1", "p2test-2", "p2test-3"]},
    "smoke": {"inits": range(60, 70), "seeds": ["smoke-1"]},
}


def episode_list(name: str, tasks=None) -> list[tuple[int, int, str]]:
    s = SETS[name]
    return [(t, i, sd) for t in (tasks if tasks is not None else range(10)) for i in s["inits"] for sd in s["seeds"]]


def load_config(path: str | None) -> dict:
    import yaml
    return yaml.safe_load(open(path)) if path else {}


def build_arms(spec_list: list[dict]):
    from second_look.loop import ArmSpec
    return [ArmSpec(**s) for s in spec_list]


def make_cfg(name: str, task: int, init: int, seed_id: str, conf: dict):
    from second_look.cusum import CusumParams
    from second_look.loop import RunCfg
    a = conf.get("accel", {"mu0": 0.0, "sigma": 1.0, "h": 1e12})
    scale = np.asarray(conf["accel_scale"], np.float32) if conf.get("accel_scale") else np.ones(70, np.float32)
    st = conf.get("stac")
    thr = {(int(sd), op): float(v) for sd, ops in (conf.get("thresholds") or conf.get("deltadino", {}).get("thresholds_val_label_v2", {})).items() for op, v in ops.items()}
    for sd, ops in (conf.get("gate_cal", {}).get("thresholds") or {}).items():            # configs/part2_gate_cal.yaml: per-target thresholds (fpr05, fpr03)
        for op, per_target in ops.items():
            for tg, v in per_target.items():
                thr[(int(sd), op, tg)] = float(v)
    haz = np.asarray(conf["p5_hazard"], np.float64) if conf.get("p5_hazard") is not None else None
    return RunCfg(name, task, init, seed_id, CusumParams.make(a["mu0"], a["sigma"], a["h"]), scale,
                  stac=CusumParams.make(st["mu0"], st["sigma"], st["h"]) if st else (CusumParams.make(0.0, 1.0, 1e12) if name == "cal" else None),
                  thresholds=thr, p5_hazard=haz, trace=False, keep_raw=(name == "cal"))


def check_thresholds(arms, conf: dict) -> None:
    """Every arm that consults the gate (P3, P6) must find its threshold for every target it uses: a missing one would silently mean "UNCHANGED"."""
    cfg = make_cfg("dev", 0, 0, "check", conf)
    for a in arms:
        if a.kind in ("P3", "P6"):
            if cfg.thresholds.get((a.seed, a.op, a.target), cfg.thresholds.get((a.seed, a.op))) is None:
                raise SystemExit(f"arm {a.name}: no gate threshold for (seed {a.seed}, op {a.op!r}, target {a.target}); pass --gate-cal configs/part2_gate_cal.yaml")


def summarise(res: dict) -> dict:
    out = {}
    for arm, r in res.items():
        out[arm] = {k: v for k, v in r.items() if k not in ("log", "pairs", "raw", "branches")}
        out[arm]["decisions_log"] = r["log"]["decisions"]
        out[arm]["events"] = r["log"]["events"]
    return out


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--config", default=None, help="configs/part2_frozen.yaml (detector parameters, P5 hazard, DeltaDINO thresholds)")
    ap.add_argument("--gate-cal", default=None, help="configs/part2_gate_cal.yaml (per-target gate thresholds fpr05 / fpr03, seeds 0-2); required for dev and test")
    ap.add_argument("--arms", default=None, help="yaml with `arms: {dev: [...], test: [...]}` (configs/part2_dev_grid.yaml, configs/part2_test_arms.yaml)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes of this worker's share (pipeline checks)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    assert 1 <= a.workers <= 2 and 0 <= a.worker < a.workers, "at most 2 workers"
    sys.path.insert(0, str(ROOT))
    from second_look import p2guard
    if a.set == "dev":
        print("DEV allowed; analysis plan committed in", p2guard.check_dev())
    if a.set == "test":
        p2guard.check_test()
    conf = load_config(a.config)
    if a.set in ("dev", "test"):
        if a.gate_cal != "configs/part2_gate_cal.yaml":
            raise SystemExit(f"{a.set.upper()} refused: pass --gate-cal configs/part2_gate_cal.yaml (the committed in-loop gate thresholds)")
    if a.gate_cal:
        conf["gate_cal"] = load_config(a.gate_cal)
    if a.arms:
        conf["arms"] = load_config(a.arms)["arms"]
    arms_key = a.set if a.set in conf.get("arms", {}) else "smoke"
    arms = build_arms(conf.get("arms", {}).get(arms_key, [])) if a.set != "cal" else []
    check_thresholds(arms, conf)
    out = Path(a.out) if a.out else ROOT / "results" / "part2" / a.set
    out.mkdir(parents=True, exist_ok=True)
    eps = episode_list(a.set, a.tasks)[a.worker::a.workers]
    if a.limit:
        eps = eps[: a.limit]
    todo = [e for e in eps if not (out / f"ep_{e[0]}_{e[1]}_{e[2]}.json.gz").exists()]
    print(f"{a.set}: {len(eps)} episodes in this share, {len(todo)} to do, {len(arms)} branched arms, out={out}", flush=True)
    from second_look.loop import Rig, run_episode
    rig = Rig()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t_start = time.time()
    for n, (task, init, seed_id) in enumerate(todo):
        t0 = time.time()
        cfg = make_cfg(a.set, task, init, seed_id, conf)
        res = run_episode(rig, cfg, arms)
        summ = summarise(res)
        rec = {"set": a.set, "task": task, "init": init, "seed_id": seed_id, "git": git, "secs": time.time() - t0, "arms": summ}
        if a.set == "cal":                                   # executed-window denoise paths + STAC scores for the offline fit
            np.savez_compressed(out / f"ep_{task}_{init}_{seed_id}.npz.tmp.npz", windows=np.stack([x[0][:, :10, :] for x in res["P0"]["raw"]]).astype(np.float32))
            os.replace(out / f"ep_{task}_{init}_{seed_id}.npz.tmp.npz", out / f"ep_{task}_{init}_{seed_id}.npz")
        else:                                                 # gate pairs, de-duplicated by content (arms sharing an alarm and a target produce identical frames)
            import hashlib
            pairs, index = {}, {}
            for arm, r in res.items():
                index[arm] = []
                for pr in r["pairs"]:
                    hs = []
                    for x in pr:
                        h = hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=8).hexdigest()
                        pairs[h] = x
                        hs.append(h)
                    index[arm].append(hs)
            rec["pair_index"] = index                         # arm -> [[ref_hash, set_hash], ...] in the order of the arm's gate events ("pair" field); arrays in the .npz by hash
            if pairs:
                np.savez_compressed(out / f"ep_{task}_{init}_{seed_id}.npz.tmp.npz", **pairs)
                os.replace(out / f"ep_{task}_{init}_{seed_id}.npz.tmp.npz", out / f"ep_{task}_{init}_{seed_id}.npz")
        write_atomic(out / f"ep_{task}_{init}_{seed_id}.json.gz", gzip.compress(json.dumps(rec, default=float).encode()))     # written last: its presence = episode finished
        done = n + 1
        print(f"[{done}/{len(todo)}] task {task} init {init} {seed_id}: {time.time() - t0:.0f}s  P0 success={res['P0']['success']}  "
              f"arms: " + " ".join(f"{k}={'S' if v['success'] else ('F' if v['flagged'] else '-')}" for k, v in res.items() if k != 'P0') + f"  (eta {((time.time() - t_start) / done) * (len(todo) - done) / 60:.0f} min)", flush=True)
    rig.close()


if __name__ == "__main__":
    main()
