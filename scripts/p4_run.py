"""Part 4 runner (SPEC_part4.md): the Part 2 loop with the arms of configs/part4_arms.yaml (+ the model arms for TEST). Resumable, atomic per-episode files, one worker.

  scripts/safe_run.sh python scripts/p4_run.py --set bench          # wall-time bench: init 20-22, seed p4bench-1, the 9 GEN arms (results never used)
  scripts/safe_run.sh python scripts/p4_run.py --set gen            # GEN: tasks 0-9 x init 0-19 x p4gen-1..3 (600 episodes; --seeds p4gen-4 adds a seed): P0 + 4 retry + 4 continue branches per alarm case
  scripts/safe_run.sh python scripts/p4_run.py --set dev            # DEV: tasks 0-9 x init 40-49 x p4dev-1,2 (200 episodes): P0, P2r, PC (the SH variants are exact selections between P2r and PC, see p4_dev_decide.py)
  scripts/safe_run.sh python scripts/p4_run.py --set test           # TEST: tasks 0-9 x init 20-39 x p4test-1..5 (1000 episodes): P0, P2r, PC, SH, SH-task, SH-noG, SH+F (delta, tau from the DEV decision)
Output: results/part4/<set>/ep_<task>_<init>_<seed>.json.gz. The split label of the noise keys is the set name (p4gen, p4dev, p4test, p4bench): no key coincides with Parts 2-3.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_run  # noqa: E402
from second_look.scene import ROOT  # noqa: E402

SETS = {
    "bench": {"inits": [20, 21, 22], "seeds": ["p4bench-1"]},
    "gen": {"inits": list(range(0, 20)), "seeds": [f"p4gen-{i}" for i in range(1, 4)]},          # 3 seeds (600 episodes): the bench projected > 9 h for 4 seeds; p4gen-4 / -5 are added with --seeds if fewer than 200 cases result
    "dev": {"inits": list(range(40, 50)), "seeds": ["p4dev-1", "p4dev-2"]},
    "test": {"inits": list(range(20, 40)), "seeds": [f"p4test-{i}" for i in range(1, 6)]},
}
SPLIT = {k: f"p4{k}" for k in SETS}
MODEL, MODEL_JSON, DECISION = ROOT / "configs/part4_model.pkl", ROOT / "configs/part4_model.json", ROOT / "results/part4/notes/p4_dev_decision.json"


def episode_list(name: str, tasks=None, seeds=None):
    s = SETS[name]
    return [(t, i, sd) for sd in (seeds if seeds is not None else s["seeds"]) for t in (tasks if tasks is not None else range(10)) for i in s["inits"]]


def load_models():
    rec = json.load(open(MODEL_JSON))
    blob = MODEL.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == rec["sha256"], "configs/part4_model.pkl does not match its recorded sha256"
    return pickle.loads(blob)


def test_arms(decision: dict):
    from second_look.loop import ArmSpec
    d = decision["delta"]
    arms = [ArmSpec("P0", "P0"), ArmSpec("P2r", "SH", "accel", target="L6", N=1, op="fpr05", mode="retry"), ArmSpec("PC", "SH", "accel", target="L6", N=1, op="fpr05", mode="continue")]
    for name, model in (("SH", "sh"), ("SH-task", "task"), ("SH-noG", "noG")):
        arms.append(ArmSpec(name, "SH", "accel", target="L6", N=1, op="fpr05", mode="model", model=model, delta=float(d[model])))
    if decision.get("tau") is not None:
        arms.append(ArmSpec("SH+F", "SH", "accel", target="L6", N=1, op="fpr05", mode="model", model="sh", delta=float(d["sh"]), tau=float(decision["tau"])))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--config", default="configs/part2_frozen.yaml")
    ap.add_argument("--gate-cal", default="configs/part2_gate_cal.yaml")
    ap.add_argument("--arms", default="configs/part4_arms.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from second_look import p4guard
    if a.set in ("gen", "dev", "test"):
        if (a.config, a.gate_cal, a.arms) != ("configs/part2_frozen.yaml", "configs/part2_gate_cal.yaml", "configs/part4_arms.yaml"):
            raise SystemExit(f"{a.set.upper()} refused: the committed Part 2 detector config, gate thresholds and configs/part4_arms.yaml must be used unchanged")
        {"gen": p4guard.check_gen, "dev": p4guard.check_dev, "test": p4guard.check_test}[a.set]()
        print(f"{a.set.upper()} allowed: the pre-registration and every record it needs are committed and clean")
    conf = p2_run.load_config(a.config)
    conf["gate_cal"] = p2_run.load_config(a.gate_cal)
    ay = p2_run.load_config(a.arms)["arms"]
    models = {}
    if a.set == "test":
        decision = json.load(open(DECISION))
        assert decision.get("go") is True, "the DEV decision says no-go: TEST must not run"
        arms = test_arms(decision)
        models = load_models()
    else:
        arms = p2_run.build_arms(ay[a.set if a.set in ay else "bench"])
        if a.set == "dev":
            models = load_models()                                   # DEV logs the model's p-hats next to the features (decision made offline from P2r / PC)
    out = Path(a.out) if a.out else ROOT / "results" / "part4" / a.set
    out.mkdir(parents=True, exist_ok=True)
    eps = episode_list(a.set, a.tasks, a.seeds)
    if a.limit:
        eps = eps[: a.limit]
    todo = [e for e in eps if not (out / f"ep_{e[0]}_{e[1]}_{e[2]}.json.gz").exists()]
    print(f"{a.set}: {len(eps)} episodes, {len(todo)} to do, arms {', '.join(x.name for x in arms)}, out={out}", flush=True)
    from second_look.loop import Rig, run_episode
    rig = Rig()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t_start = time.time()
    for n, (task, init, seed_id) in enumerate(todo):
        t0 = time.time()
        cfg = dataclasses.replace(p2_run.make_cfg(SPLIT[a.set], task, init, seed_id, conf), p4=True, sh_models=models)
        res = run_episode(rig, cfg, arms)
        summ = p2_run.summarise(res)
        rec = {"set": a.set, "task": task, "init": init, "seed_id": seed_id, "git": git, "secs": time.time() - t0, "arms": summ}
        p2_run.write_atomic(out / f"ep_{task}_{init}_{seed_id}.json.gz", gzip.compress(json.dumps(rec, default=float).encode()))
        print(f"[{n + 1}/{len(todo)}] task {task} init {init} {seed_id}: {time.time() - t0:.0f}s  P0 success={res['P0']['success']}  arms: "
              + " ".join(f"{k}={'S' if v['success'] else ('F' if v['flagged'] else '-')}" for k, v in res.items() if k != "P0")
              + f"  (eta {((time.time() - t_start) / (n + 1)) * (len(todo) - n - 1) / 60:.0f} min)", flush=True)
    rig.close()


if __name__ == "__main__":
    main()
