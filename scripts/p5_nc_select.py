"""P5.0 step 2: the alarm states for the noise-ceiling test, from the Part 4 GEN logs (init 0-19). Plan: results/part5/notes/p5_0_plan.txt.
  python scripts/p5_nc_select.py       -> results/part5/nc/states.json
Population A: P0-failed alarmed episodes, quota min(count, 8) per task, the largest tasks lose one each until 60; C: P0-succeeded, up to 20 by the same rule. Sampling: numpy default_rng(0) over the sorted (task, init, seed) list."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C

ROOT = C.ROOT
CAP, TOTAL_A, TOTAL_C = 8, 60, 20


def quotas(counts: dict, cap: int, total: int) -> dict:
    q = {t: min(n, cap) for t, n in counts.items()}
    while sum(q.values()) > total:
        t = max(sorted(q), key=lambda t: (q[t], counts[t]))          # the task with the largest quota (ties: most cases) loses one
        q[t] -= 1
    return q


def pick(cases: list[dict], cap: int, total: int, rng) -> list[dict]:
    by = defaultdict(list)
    for c in sorted(cases, key=lambda c: c["key"]):
        by[c["task"]].append(c)
    q = quotas({t: len(v) for t, v in by.items()}, cap, total)
    out = []
    for t in sorted(by):
        idx = rng.choice(len(by[t]), size=q[t], replace=False)
        out += [by[t][i] for i in sorted(idx)]
    return out


def main() -> None:
    recs = C.load("results/part4/gen")
    cases, skipped = C.gen_cases(recs)
    rng = np.random.default_rng(0)
    A = pick([c for c in cases if not c["p0"]], CAP, TOTAL_A, rng)
    Cc = pick([c for c in cases if c["p0"]], CAP, TOTAL_C, rng)
    ctrl = {(a["task"], a["init"], a["seed"]) for a in A[:: max(1, len(A) // 5)][:5]}          # 5 control states: every 12th state of A
    states = []
    for pop, lst in (("A", A), ("C", Cc)):
        for c in lst:
            assert 0 <= c["init"] < 20
            states.append({"pop": pop, "task": c["task"], "init": c["init"], "seed_id": c["seed"], "d": c["d"], "elapsed": c["elapsed"], "scene": c["truth"], "p0_success": c["p0"],
                           "control": (c["task"], c["init"], c["seed"]) in ctrl})
    out = ROOT / "results" / "part5" / "nc"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "results/part4/gen", "skipped": skipped, "states": states}, open(out / "states.json", "w"), indent=1)
    for pop in "AC":
        s = [x for x in states if x["pop"] == pop]
        print(pop, len(s), "per task", {t: sum(1 for x in s if x["task"] == t) for t in sorted({x["task"] for x in s})}, "scene", {k: sum(1 for x in s if x["scene"] == k) for k in sorted({x["scene"] for x in s})})
    print("controls", sum(x["control"] for x in states))


if __name__ == "__main__":
    main()
