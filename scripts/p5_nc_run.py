"""P5.0 runner: the noise-ceiling branches (plan: results/part5/notes/p5_0_plan.txt). Resumable, atomic per-state files, one worker.

  scripts/safe_run.sh python scripts/p5_nc_run.py --bench                # 3 states, K small, keys p5nc1000+ (never used in results): wall-time projection
  scripts/safe_run.sh python scripts/p5_nc_run.py                        # the full run from configs/part5_noise_ceiling.yaml (needs the plan + yaml committed)
Per state: replay the Part 4 GEN P0 episode (split p4gen, same keys), assert its decisions equal the GEN log bit for bit, branch at the first accel alarm into K_R retry branches (Part 4's P2r path with the
namespace "p5nc") and K_C continue branches (A only: re-plan at the alarm, no pause, no rewind); control states also rerun GEN's own R1-R4 / C1-C4 and compare.
Output: results/part5/nc/states/s_<pop>_<task>_<init>_<seed>.json.gz (bench: results/part5/bench)."""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_run  # noqa: E402
import p4_common as C  # noqa: E402
from second_look.scene import ROOT  # noqa: E402

SPLIT = "p4gen"
OUT = ROOT / "results" / "part5" / "nc"


def order(states: list[dict]) -> list[dict]:
    """Round-robin over the tasks inside each population (A first, then C), so any prefix is stratified."""
    out = []
    for pop in "AC":
        by: dict = {}
        for s in states:
            if s["pop"] == pop:
                by.setdefault(s["task"], []).append(s)
        while any(by.values()):
            for t in sorted(by):
                if by[t]:
                    out.append(by[t].pop(0))
    return out


def arms_for(state: dict, k_r: int, k_c: int, key0: int, controls: bool):
    from second_look.loop import ArmSpec
    arms = [ArmSpec(f"R{j}", "SH", "accel", target="L6", N=1, op="fpr05", mode="retry", ns="p5nc", key=key0 + j) for j in range(k_r)]
    arms += [ArmSpec(f"C{j}", "SH", "accel", target="L6", N=1, op="fpr05", mode="cont_nopause", ns="p5nc", key=key0 + j) for j in range(k_c)]
    if controls:
        gen = p2_run.build_arms(p2_run.load_config("configs/part4_arms.yaml")["arms"]["gen"])
        arms += [dataclasses.replace(a, name="G" + a.name) for a in gen if a.kind == "SH"]          # GEN's own branches: GR1..GR4, GC1..GC4
    return arms


def branch_row(r: dict, ev_d: int) -> dict:
    dl = r["decisions_log"]
    after = [x for x in dl if isinstance(x["iv"], str) and x["iv"].startswith("p5nc")]
    return {"success": bool(r["success"]), "elapsed": r["elapsed"], "flagged": r["flagged"], "flag_reason": r["flag_reason"], "rewinds": r["rewinds"], "invalid_rewinds": r["invalid_rewinds"],
            "cut_short": r["cut_short"], "n_decisions_after": len(after), "alarm_again": any(x["alarm_a"] for x in after), "p5": r.get("p5", {}),
            "return_cost": next((e.get("return_cost") for e in r["events"] if e.get("kind") == "gate"), None)}


def run_state(rig, conf, state: dict, k_r: int, k_c: int, key0: int, gen_rec: dict) -> dict:
    from second_look.loop import run_episode
    t0 = time.time()
    cfg = dataclasses.replace(p2_run.make_cfg(SPLIT, state["task"], state["init"], state["seed_id"], conf), p4=True)
    arms = arms_for(state, k_r, k_c, key0, state["control"])
    res = run_episode(rig, cfg, arms)
    summ = p2_run.summarise(res)
    p0, g0 = summ["P0"], gen_rec["arms"]["P0"]
    keep = ("d", "iv", "accel", "S_a", "alarm_a", "obs", "elapsed")
    same = [{k: x[k] for k in keep} for x in p0["decisions_log"]] == [{k: x[k] for k in keep} for x in g0["decisions_log"]] and p0["success"] == g0["success"] and p0["elapsed"] == g0["elapsed"]
    gate = next((e for n, a in summ.items() if n.startswith("R") for e in a["events"] if e.get("kind") == "gate"), None)
    d_ok = gate is not None and gate["d"] == state["d"]
    rows = {n: branch_row(a, state["d"]) for n, a in summ.items() if n[0] in "RC" and n != "P0"}
    ctrl = None
    if state["control"]:
        ctrl = {}
        for n in ("R1", "R2", "R3", "R4", "C1", "C2", "C3", "C4"):
            g, a = gen_rec["arms"][n], summ["G" + n]
            ctrl[n] = {"gen": [bool(g["success"]), g["elapsed"]], "now": [bool(a["success"]), a["elapsed"]], "equal": bool(g["success"]) == bool(a["success"]) and g["elapsed"] == a["elapsed"]}
    return {"state": state, "replay_equal": bool(same), "d_equal": bool(d_ok), "p0": {"success": bool(p0["success"]), "elapsed": p0["elapsed"]}, "scene_at_d": {"truth": gate and gate["truth"], "gray": gate and gate["truth_gray"]},
            "k_r": k_r, "k_c": k_c, "key0": key0, "branches": {n: r for n, r in rows.items() if not n.startswith("G")}, "controls": ctrl, "secs": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--yaml", default="configs/part5_noise_ceiling.yaml")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from second_look import p5guard
    sj = json.load(open(OUT / "states.json"))
    states = order(sj["states"])
    p5guard.check_inits(states)
    if a.bench:
        picks, seen = [], set()
        for s in states:                                                         # 3 states of 3 different tasks of A, one of them a control (checks the controls too)
            if s["pop"] == "A" and s["task"] not in seen and len(picks) < 3 and (s["task"] in (0, 3, 9)):
                picks.append(s); seen.add(s["task"])
        states, k_r_a, k_c_a, k_r_c, key0, out = picks, 8, 4, 4, 1000, ROOT / "results/part5/bench"
    else:
        p5guard.check_run()
        y = p2_run.load_config(a.yaml)
        assert hashlib.sha256((OUT / "states.json").read_bytes()).hexdigest() == y["states_sha256"], "states.json does not match the yaml"
        states = [s for s in states if s["pop"] == "C"] + [s for s in states if s["pop"] == "A"][: y["n_A"]]
        states = order(states)
        k_r_a, k_c_a, k_r_c, key0, out = y["K_AR"], y["K_AC"], y["K_CR"], 0, OUT / "states"
    out.mkdir(parents=True, exist_ok=True)
    conf = p2_run.load_config("configs/part2_frozen.yaml")
    conf["gate_cal"] = p2_run.load_config("configs/part2_gate_cal.yaml")
    recs = C.load("results/part4/gen")
    todo = [s for s in states if not (out / f"s_{s['pop']}_{s['task']}_{s['init']}_{s['seed_id']}.json.gz").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"p5.0 {'bench' if a.bench else 'full'}: {len(states)} states, {len(todo)} to do, out={out}", flush=True)
    from second_look.loop import Rig
    rig = Rig()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t_start = time.time()
    for n, s in enumerate(todo):
        k_r, k_c = (k_r_a, k_c_a) if s["pop"] == "A" else (k_r_c, 0)
        rec = run_state(rig, conf, s, k_r, k_c, key0, recs[(s["task"], s["init"], s["seed_id"])])
        rec["git"] = git
        p2_run.write_atomic(out / f"s_{s['pop']}_{s['task']}_{s['init']}_{s['seed_id']}.json.gz", gzip.compress(json.dumps(rec, default=float).encode()))
        succ = np.mean([b["success"] for k, b in rec["branches"].items() if k.startswith("R")])
        print(f"[{n + 1}/{len(todo)}] {s['pop']} task {s['task']} init {s['init']} {s['seed_id']}: {rec['secs']:.0f}s  replay_equal={rec['replay_equal']} d_equal={rec['d_equal']}  retry success {succ:.2f}"
              + (f"  controls {all(c['equal'] for c in rec['controls'].values())}" if rec["controls"] else "") + f"  (eta {((time.time() - t_start) / (n + 1)) * (len(todo) - n - 1) / 60:.0f} min)", flush=True)
    rig.close()


if __name__ == "__main__":
    main()
