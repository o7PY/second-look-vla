"""R6: model choice by the pre-written rule (SPEC_before_part2.md section 10). Evaluates candidates A (DeltaDINO-v2, arm + static mask) and C (static mask only) on
val and the dev sets (test, held-out, policy, probe). Mean over the 3 seeds; 95% CIs by bootstrap over episodes (episodes resampled jointly for the 3 seeds).
B (no masking, one seed) is reported for reference only. Writes results/part1/b2/r6.json and results/part1/b2/notes/r6_decision.md.   python scripts/b2_r6.py"""
import json
import numpy as np, pandas as pd
from second_look import metrics as M
from second_look.p15 import Split
from second_look.scene import ROOT

OUT = ROOT / "results/part1/b2"
CAND = {"A": {"prefix": "deltadino_v2_seed", "c2": "_v2s", "ckpt": "results/part1/deltadino_v2/seed{}/best.pt", "r1": "r1_A.json", "seeds": [0, 1, 2], "name": "DeltaDINO-v2 (arm mask + static mask)"},
        "C": {"prefix": "deltadino_v2static_seed", "c2": "_v2cs", "ckpt": "results/part1/deltadino_v2_static/seed{}/best.pt", "r1": "r1_C.json", "seeds": [0, 1, 2], "name": "candidate C (static mask only, no arm mask)"},
        "B": {"prefix": "deltadino_v2nomask_seed", "c2": "_v2nm", "c2_exact": True, "ckpt": "results/part1/deltadino_v2_nomask/seed{}/best.pt", "r1": "r1_B.json", "seeds": [0], "name": "B: no masking at all (reference only, not eligible)"}}
DEV = ("test", "test_heldout_tasks", "test_policy")
rng = np.random.default_rng(0)


def per_seed(c):
    r = []
    for k in c["seeds"]:
        m = c["prefix"] + str(k); row = {"seed": k, "method": m, "ckpt": c["ckpt"].format(k)}
        v = Split("val", m, labels="v2"); thr = v.thr; row["threshold_v2"] = thr
        for sp in ("val",) + DEV + ("probe_static",):
            S = Split(sp, m, labels="v2", thr=thr); ng = S.nongray; y = S.y[ng]
            big = ng & (S.y == 1) & ((S.meta.max_trans_cm.values > 2) | (S.meta.max_rot_deg.values > 15))
            h = M.hard_negative_far(S.meta, S.score, thr)["H1+H2"]["false_alarm"]
            row[sp] = {"auroc": M.auroc(y, S.score[ng]) if len(np.unique(y)) == 2 else None, "recall": float(S.pred[ng][y == 1].mean()), "unchanged_flag": float(S.pred[ng][y == 0].mean()) if (y == 0).any() else None,
                       "big_change_recall": float(S.pred[big].mean()) if big.any() else None, "h1h2_fa": None if h != h else h, "f1": M.prf(y, S.score[ng], thr)["f1"]}
        c2 = json.load(open(OUT.parent / "p15" / (f"c2_metrics{c['c2']}.json" if c.get("c2_exact") else f"c2_metrics{c['c2']}{k}.json")))   # B's file has no seed suffix
        row["armswap_pos_test"] = c2["C"]["positives_test"]["flip_rate"]; row["armswap_h1"] = c2["C"]["H1_unchanged_test"]["flip_rate"]
        row["p_adj"] = {sp: c2["A"][sp]["overall"]["P_adj"] for sp in DEV}; row["p_visible"] = {sp: c2["A"][sp]["overall"]["P_visible"] for sp in DEV}
        r.append(row)
    return r


def boot_auroc(c, split, n=500):
    """CI of the seed-mean AUROC, episodes resampled jointly for all seeds."""
    S = [Split(split, c["prefix"] + str(k), labels="v2") for k in c["seeds"]]
    ng = S[0].nongray; meta = S[0].meta[ng].reset_index(drop=True); y = S[0].y[ng]
    ep = (meta.task.astype(int) * 1000 + meta.demo.astype(int)).values; u, inv = np.unique(ep, return_inverse=True); idx = [np.flatnonzero(inv == j) for j in range(len(u))]
    sc = [s.score[ng] for s in S]; vals = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.integers(len(u), size=len(u))])
        if len(np.unique(y[ii])) < 2: continue
        vals.append(np.mean([M.auroc(y[ii], s[ii]) for s in sc]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


res = {}
for key, c in CAND.items():
    ps = per_seed(c); r1 = json.load(open(OUT / c["r1"]))
    mean = lambda f: float(np.mean([f(p) for p in ps]))
    agg = {"pooled_lt20_removal_drop": float(np.mean([r1[p["method"]]["pooled_lt20"]["drop"] for p in ps])),
           "policy_removal_drop": float(np.mean([r1[p["method"]]["overall"]["test_policy"]["drop"] for p in ps])),
           "armswap_pos_test": mean(lambda p: p["armswap_pos_test"]), "armswap_h1": mean(lambda p: p["armswap_h1"]),
           "test_h1h2_fa": mean(lambda p: p["test"]["h1h2_fa"]), "test_auroc": mean(lambda p: p["test"]["auroc"]), "probe_recall": mean(lambda p: p["probe_static"]["recall"]),
           "heldout_auroc": mean(lambda p: p["test_heldout_tasks"]["auroc"]), "policy_unchanged_flag": mean(lambda p: p["test_policy"]["unchanged_flag"]),
           "policy_auroc": mean(lambda p: p["test_policy"]["auroc"]), "heldout_big_change_recall": mean(lambda p: p["test_heldout_tasks"]["big_change_recall"]),
           "pooled_lt20_n": r1[ps[0]["method"]]["pooled_lt20"]["n"]}
    gates = {"G1_removal": bool(agg["pooled_lt20_removal_drop"] >= 0.90 and agg["policy_removal_drop"] >= 0.80), "G2_arm": bool(agg["armswap_pos_test"] <= 0.05 and agg["armswap_h1"] <= 0.01),
             "G3_false_alarms": bool(agg["test_h1h2_fa"] <= 0.01), "G4_in_distribution": bool(agg["test_auroc"] >= 0.995), "G5_static_probe": bool(agg["probe_recall"] >= 0.80)}
    res[key] = {"name": c["name"], "per_seed": ps, "mean": agg, "gates": gates, "all_gates": all(gates.values()), "auroc_ci": {sp: boot_auroc(c, sp) for sp in ("test", "test_heldout_tasks", "test_policy")}}
A, C = res["A"], res["C"]
if A["all_gates"] and C["all_gates"]:
    better = C["mean"]["heldout_auroc"] >= A["mean"]["heldout_auroc"] + 0.005 and C["mean"]["policy_unchanged_flag"] <= A["mean"]["policy_unchanged_flag"] + 0.01
    choice, why = ("C" if better else "A"), "both pass all gates; " + ("C's mean held-out AUROC is at least A's + 0.005 and its policy unchanged-flag rate is not worse by more than 1 point" if better else "C does not meet both of: held-out AUROC >= A + 0.005 and policy unchanged-flag not worse than A by more than 1 point -> A (simpler story, already validated)")
elif A["all_gates"] != C["all_gates"]:
    choice = "A" if A["all_gates"] else "C"; why = f"only {choice} passes all gates"
else:
    choice = "A"; why = "neither passes all gates -> A, with the failed gates listed as known limitations; Part 2 proceeds in-distribution only"
res["choice"] = {"model": choice, "reason": why}
json.dump(res, open(OUT / "r6.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

f = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"
G = {"G1_removal": "removal: pooled < 20% stratum drop ≥ 0.90 and policy overall drop ≥ 0.80", "G2_arm": "arm swap: flips ≤ 5% (positives, test) and ≤ 1% (H1)", "G3_false_alarms": "test H1+H2 false alarms ≤ 1%",
     "G4_in_distribution": "test AUROC ≥ 0.995", "G5_static_probe": "probe recall ≥ 0.80"}
L = ["# R6: model choice (rule written in `SPEC_before_part2.md` §10 before any candidate-C result existed)", "",
     f"## Outcome: **{choice}** ({res[choice]['name']})", "", f"Reason: {why}.", "",
     "Evaluated on val and the dev sets (test, held-out, policy, R2-B probe), mean over 3 seeds, each seed at its own v2 val-F1-max threshold. **`test_policy2` has not been opened** (this note must be committed before R7; `src/second_look/sealed.py` enforces it). B (one seed, no masks) is reference only.", "",
     "## Gates", "", "| gate | requirement | A | C | B (reference) |", "|---|---|---|---|---|"]
vals = lambda k, key: {"G1_removal": f"{f(res[k]['mean']['pooled_lt20_removal_drop'])} / {f(res[k]['mean']['policy_removal_drop'])}", "G2_arm": f"{f(res[k]['mean']['armswap_pos_test'])} / {f(res[k]['mean']['armswap_h1'])}",
                       "G3_false_alarms": f(res[k]["mean"]["test_h1h2_fa"]), "G4_in_distribution": f(res[k]["mean"]["test_auroc"]), "G5_static_probe": f(res[k]["mean"]["probe_recall"])}[key]
for g, req in G.items():
    L.append(f"| {g} | {req} | {vals('A', g)} {'✅' if A['gates'][g] else '❌'} | {vals('C', g)} {'✅' if C['gates'][g] else '❌'} | {vals('B', g)} {'✅' if res['B']['gates'][g] else '❌'} |")
L += ["", f"(G1 shows pooled < 20%-stratum drop / policy overall drop; the pooled stratum holds n = {A['mean']['pooled_lt20_n']} (A, seed 0) / {C['mean']['pooled_lt20_n']} (C, seed 0) pairs. G2 shows positives / H1.)", "",
      "## Tie-break quantities (used only when both pass all gates)", "", "| | A | C | B (ref.) |", "|---|---|---|---|"]
for lab, key in (("held-out AUROC (mean of seeds)", "heldout_auroc"), ("policy unchanged-flag rate", "policy_unchanged_flag"), ("policy AUROC", "policy_auroc"), ("held-out big-change recall", "heldout_big_change_recall")):
    L.append(f"| {lab} | {f(A['mean'][key])} | {f(C['mean'][key])} | {f(res['B']['mean'][key])} |")
L += ["", "## Per model: AUROC (95% CI by bootstrap over episodes, seed mean) on the dev sets", "", "| model | test | held-out tasks | policy |", "|---|---|---|---|"]
for k in ("A", "C", "B"):
    ci = res[k].get("auroc_ci", {})
    L.append(f"| {k} | " + " | ".join(f"{f(np.mean([p[sp]['auroc'] for p in res[k]['per_seed']]))} [{f(ci[sp][0])}, {f(ci[sp][1])}]" for sp in DEV) + " |")
L += ["", "## Chosen checkpoints and thresholds (logit, chosen on val with `label_v2`, one per checkpoint, never copied)", "", "| seed | checkpoint | F1-max threshold |", "|---|---|---|"]
for p in res[choice]["per_seed"]: L.append(f"| {p['seed']} | `{p['ckpt']}` | {p['threshold_v2']:.3f} |")
L += ["", "The operating-point table (thresholds at val recall 0.95 / 0.98 / 0.99 / 0.995 and F1-max) for the chosen model is in `results/part1/b2/notes/r6_operating_points.md`; no threshold is chosen here."]
nar = ROOT / "results/part1/b2/notes/r6_narrative.md"
(ROOT / "results/part1/b2/notes/r6_decision.md").write_text("\n".join(L) + "\n" + (nar.read_text() if nar.exists() else "")); print("\n".join(L[:30]))
