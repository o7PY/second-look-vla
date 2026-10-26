"""P4.8: TEST analysis, exactly as pre-registered (results/part4/notes/p4_analysis_plan.txt; SPEC_part4.md sections 8-9).

  python scripts/p4_analyse.py [--dir results/part4/test] [--boot 10000]     -> results/part4/results_test.json, results/part4/notes/p4_test_results.md

Unit = episode (task, init, seed); cluster = (task, init) (the 5 seeds of an init state are one cluster); cluster bootstrap B = 10,000, percentile 95% CIs, one-sided p floored at 1/(B+1), Holm over H1-H3, exact McNemar.
All 10 tasks are primary. Arms absent from an episode's log never alarmed: they equal P0.
  H1  S(SH) - S(P0) > 0      H2  S(SH) - S(P2r) > 0      H3  S(SH) - S(SH-task) > 0     (p = share of resamples with the difference <= 0)
Integrity check (asserted, reported): every SH arm equals the physical branch it chose (retry == P2r, continue == PC) in outcome and elapsed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C
from p2_analyse import Boot, mcnemar, per_episode, pct

ROOT = C.ROOT
ARMS = ["P0", "P2r", "PC", "SH", "SH-task", "SH-noG", "SH+F"]


def load_eps(d):
    recs = C.load(d)
    eps = []
    for k, r in recs.items():
        arms = {a: C.arm_of(r, a) for a in ARMS}
        eps.append({"task": k[0], "init": k[1], "seed_id": k[2], "arms": arms, "raw": r})
    return eps, recs


def table(eps, B, arms):
    if not eps:
        return {}
    bt = Boot(eps, B)
    out = {"n_episodes": len(eps), "n_clusters": bt.C, "arms": {}}
    for arm in arms:
        c = per_episode(eps, arm)
        row = {}
        for k in ("S", "SF", "F", "FF", "rescue", "harm"):
            est, ci, _ = bt.rate(c[k])
            row[k] = {"est": est, "ci": ci, "count": int(c[k].sum())}
        est, ci, _ = bt.rate(c["flag_true"], c["F"])
        row["FP"] = {"est": est, "ci": ci, "flags": int(c["F"].sum()), "flag_true": int(c["flag_true"].sum())}
        row["retry_rate"] = float(np.mean([bool(e["arms"][arm].get("rewinds", 0)) for e in eps]))
        row["mean_elapsed"] = float(c["elapsed"].mean())
        out["arms"][arm] = row
    return out


def hypotheses(eps, B):
    bt = Boot(eps, B)
    f = lambda a: per_episode(eps, a)
    floor = 1.0 / (B + 1)
    h = {}
    for nm, (a, b, claim) in {"H1": ("SH", "P0", "S(SH) > S(P0)"), "H2": ("SH", "P2r", "S(SH) > S(P2r)"), "H3": ("SH", "SH-task", "S(SH) > S(SH-task)")}.items():
        est, bs = bt.diff(f(a)["S"], f(b)["S"])
        h[nm] = {"claim": claim, "est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(max(np.mean(bs <= 0), floor)), "mcnemar": mcnemar(f(a)["S"], f(b)["S"])}
    order = sorted(h, key=lambda k: h[k]["p"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (3 - i) * h[k]["p"]))
        h[k]["p_holm"], h[k]["supported"] = run, bool(run < 0.05)
    d = {}
    for nm, (a, b) in {"SH - SH-noG (the camera)": ("SH", "SH-noG"), "P2r - P0": ("P2r", "P0"), "PC - P0 (the pause alone)": ("PC", "P0"), "SH-task - P0": ("SH-task", "P0"), "SH - PC": ("SH", "PC")}.items():
        est, bs = bt.diff(f(a)["S"], f(b)["S"])
        d[nm] = {"est": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "mcnemar": mcnemar(f(a)["S"], f(b)["S"])}
    h["descriptive"] = d
    return h


def integrity(eps):
    bad, n = [], 0
    for e in eps:
        raw = e["raw"]["arms"]
        for nm in ("SH", "SH-task", "SH-noG", "SH+F"):
            if nm not in raw:
                continue
            ev = C.sh_event(raw[nm])
            if ev is None:
                continue
            n += 1
            ref = {"retry": "P2r", "continue": "PC"}.get(ev["decision"])
            if ref is None:
                if not (raw[nm]["flagged"] and raw[nm]["flag_reason"] == "low_value"):
                    bad.append((e["task"], e["init"], e["seed_id"], nm, "flag"))
                continue
            a, b = raw[nm], C.arm_of(e["raw"], ref)
            if (a["success"], a["flagged"], a["elapsed"], a["rewinds"]) != (b["success"], b["flagged"], b["elapsed"], b["rewinds"]):
                bad.append((e["task"], e["init"], e["seed_id"], nm, ref))
    return {"checked": n, "mismatches": bad}


def calibration(eps):
    """The model's p_retry / p_cont against what P2r / PC actually did on the same alarmed episodes (from the SH arm's logged p-hats)."""
    rows = []
    for e in eps:
        raw = e["raw"]["arms"]
        ev = C.sh_event(raw["SH"]) if "SH" in raw else None
        if ev is None or ev["p_retry"] is None:
            continue
        rows.append((ev["p_retry"], ev["p_cont"], bool(raw["P2r"]["success"]), bool(raw["PC"]["success"])))
    if not rows:
        return {}
    a = np.array(rows, float)
    out = {"n": len(a), "mean_p_retry": float(a[:, 0].mean()), "P2r_success": float(a[:, 2].mean()), "mean_p_cont": float(a[:, 1].mean()), "PC_success": float(a[:, 3].mean()), "bins": []}
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        mc = (a[:, 1] >= lo) & (a[:, 1] < hi)
        out["bins"].append({"bin": [lo, min(hi, 1.0)], "retry_head": [int(m.sum()), float(a[m, 2].mean()) if m.any() else None, float(a[m, 0].mean()) if m.any() else None],
                            "continue_head": [int(mc.sum()), float(a[mc, 3].mean()) if mc.any() else None, float(a[mc, 1].mean()) if mc.any() else None]})
    return out


def per_task(eps):
    out = {}
    for t in range(10):
        es = [e for e in eps if e["task"] == t]
        if es:
            out[str(t)] = {"n": len(es), **{a: {"S": float(np.mean([bool(e["arms"][a]["success"]) for e in es]))} for a in ("P0", "P2r", "PC", "SH", "SH-task", "SH-noG", "SH+F")}}
    return out


def md_table(title, tb, arms):
    if not tb:
        return [f"### {title}", "", "(no episodes)", ""]
    L = [f"### {title} ({tb['n_episodes']} episodes, {tb['n_clusters']} clusters)", "", "| arm | S % [95% CI] | SF % | F % | FF % | FP % | rescues | harms | retry rate |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        if arm not in tb["arms"]:
            continue
        r = tb["arms"][arm]
        ci = lambda c: "" if not c["ci"] else f" [{pct(c['ci'][0])}, {pct(c['ci'][1])}]"
        L.append(f"| {arm} | {pct(r['S']['est'])}{ci(r['S'])} | {pct(r['SF']['est'])} | {pct(r['F']['est'])} | {pct(r['FF']['est'])} | {pct(r['FP']['est'])} ({r['FP']['flag_true']}/{r['FP']['flags']}) | {r['rescue']['count']} | {r['harm']['count']} | {100 * r['retry_rate']:.0f}% |")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/part4/test")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    eps, recs = load_eps(a.dir)
    present = [x for x in ARMS if any(x in e["raw"]["arms"] for e in eps) or x == "P0"]
    seen = [e for e in eps if e["task"] not in C.HELD]
    held = [e for e in eps if e["task"] in C.HELD]
    B = a.boot
    res = {"n_episodes": len(eps), "arms": present, "tables": {"all": table(eps, B, present), "seen": table(seen, B, present), "held_out": table(held, B, present)}, "hypotheses": {"all": hypotheses(eps, B), "seen_descriptive": hypotheses(seen, B)},
           "integrity": integrity(eps), "calibration": calibration(eps), "per_task": per_task(eps), "git": sorted({r["git"] for r in recs.values()})}
    assert not res["integrity"]["mismatches"], res["integrity"]["mismatches"]
    json.dump(res, open(ROOT / "results/part4/results_test.json", "w"), indent=1, default=float)
    h = res["hypotheses"]["all"]
    L = [f"# Part 4 TEST results ({len(eps)} episodes, all 10 tasks primary; commits {', '.join(res['git'])}; bootstrap B = {B})", ""] + md_table("All 10 tasks (primary)", res["tables"]["all"], present)
    L += ["## Hypotheses (all 10 tasks, Holm over H1-H3)", "", "| hypothesis | estimate [95% CI] | p | Holm p | supported |", "|---|---|---|---|---|"]
    for k in ("H1", "H2", "H3"):
        L.append(f"| {k} {h[k]['claim']} | {100 * h[k]['est']:+.1f} pp [{100 * h[k]['ci95'][0]:+.1f}, {100 * h[k]['ci95'][1]:+.1f}] | {h[k]['p']:.4f} | {h[k]['p_holm']:.4f} | {h[k]['supported']} |")
    L += ["", "Descriptive differences (S, pp, 95% CI; McNemar):", ""] + [f"- {k}: {100 * v['est']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]; a-only {v['mcnemar']['a_only']}, b-only {v['mcnemar']['b_only']}, p = {v['mcnemar']['p']:.3f}" for k, v in h["descriptive"].items()]
    L += ["", "## Secondary breakdowns", ""] + md_table("Seen-8", res["tables"]["seen"], present) + md_table("Tasks 6 and 8", res["tables"]["held_out"], present)
    L += ["## Integrity", "", f"SH arms equal the branch they chose: {res['integrity']['checked'] - len(res['integrity']['mismatches'])} of {res['integrity']['checked']}.", "", "## Model calibration on TEST (alarmed episodes with a decision)", "",
          json.dumps(res["calibration"], default=float), "", "## Per task (S %)", "", "| task | n | P0 | P2r | PC | SH | SH-task | SH-noG |", "|---|---|---|---|---|---|---|---|"]
    for t, v in res["per_task"].items():
        L.append(f"| {t} | {v['n']} | " + " | ".join(f"{100 * v[x]['S']:.1f}" for x in ("P0", "P2r", "PC", "SH", "SH-task", "SH-noG")) + " |")
    (ROOT / "results/part4/notes/p4_test_results.md").write_text("\n".join(L))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
