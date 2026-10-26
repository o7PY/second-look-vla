"""P5.0 step 1: free preview from the Part 4 GEN logs (descriptive, no decision). Per alarm case: the 4 retry (R1-R4) and 4 continue (C1-C4) draws.
  python scripts/p5_nc_preview.py    -> results/part5/notes/p5_0_preview.md, results/part5/nc/preview.json"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C

ROOT = C.ROOT


def block(cases: list[dict]) -> dict:
    if not cases:
        return {"n": 0}
    pr = np.array([c["y_r"].mean() for c in cases]); pc = np.array([c["y_c"].mean() for c in cases])
    ar = np.array([c["y_r"].max() for c in cases]); ac = np.array([c["y_c"].max() for c in cases])
    ae = np.array([max(c["y_r"].max(), c["y_c"].max()) for c in cases])
    return {"n": len(cases), "retry_share": pr.mean(), "cont_share": pc.mean(), "any4_retry": ar.mean(), "any4_cont": ac.mean(), "any4_either": ae.mean(),
            "gap_retry": ar.mean() - pr.mean(), "gap_cont": ac.mean() - pc.mean()}


def row(name: str, b: dict) -> str:
    if not b["n"]:
        return f"| {name} | 0 | | | | | | |"
    f = lambda k: f"{100 * b[k]:.1f}"
    return f"| {name} | {b['n']} | {f('retry_share')} | {f('any4_retry')} | {f('cont_share')} | {f('any4_cont')} | {f('any4_either')} | {f('gap_retry')} |"


def main() -> None:
    recs = C.load("results/part4/gen")
    cases, skipped = C.gen_cases(recs)
    H = "| group | cases | retry: mean p | retry: any of 4 | continue: mean p | continue: any of 4 | either: any of 8 | retry gap (any4 - p), pp |\n|---|---|---|---|---|---|---|---|"
    L = ["# P5.0 preview from the Part 4 GEN logs (descriptive; no decision; written after the plan commit a079884)", "",
         f"{len(cases)} alarm cases (first alarm reached the decision; skipped {skipped}), 4 retry draws (R1-R4, key 1..4 of the Part 2 retry) and 4 continue draws (C1-C4: pause, then fresh noise after the alarm decision) each.",
         "any-of-4 has a max-over-draws bias: with 4 draws the 'gap' is positive even when every draw has the same success probability (for p = 0.28 the expected any-of-4 is 0.73 if the draws were independent per state with p fixed; the table is what was measured, not a ceiling).", "", H]
    out = {"all": block(cases)}
    L.append(row("all cases", out["all"]))
    for nm, sel in (("P0 failed", lambda c: not c["p0"]), ("P0 succeeded", lambda c: c["p0"])):
        out[nm] = block([c for c in cases if sel(c)])
        L.append(row(nm, out[nm]))
    for sc in ("unchanged", "changed", "gray"):
        out["scene_" + sc] = block([c for c in cases if c["truth"] == sc and not c["p0"]])
        L.append(row(f"P0 failed, scene {sc}", out["scene_" + sc]))
    L += ["", "## By task (P0-failed cases / all cases)", "", H]
    for t in range(10):
        a = block([c for c in cases if c["task"] == t and not c["p0"]]); b = block([c for c in cases if c["task"] == t])
        out[f"task{t}_failed"], out[f"task{t}_all"] = a, b
        L.append(row(f"task {t}, P0 failed", a)); L.append(row(f"task {t}, all", b))
    pf = [c for c in cases if not c["p0"]]
    dist = np.array([c["y_r"].sum() for c in pf])
    L += ["", "## Distribution of the number of successes among the 4 retry draws (P0-failed cases)", "", "| successes of 4 | cases |", "|---|---|"] + [f"| {k} | {int((dist == k).sum())} |" for k in range(5)]
    L += ["", "Reading aid: a state where 1-3 of 4 draws succeed is a state where noise matters; 0 of 4 is a 'habit' candidate (or a rare success); 4 of 4 is a state where any retry works.", ""]
    (ROOT / "results" / "part5" / "notes").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/part5/notes/p5_0_preview.md").write_text("\n".join(L))
    json.dump(out, open(ROOT / "results/part5/nc/preview.json", "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
