"""P2.5: apply the pre-written DEV rule (results/part2/notes/p2_4_analysis_plan.md, "DEV rule (P2.5, verbatim)") to results/part2/dev and write the decision.

  python scripts/p2_dev_decide.py [--dev results/part2/dev] [--write]

Rule: among the P3 grid configs (target x DeltaDINO operating point x N) choose the one that maximises S on the DEV SEEN tasks (all but 6 and 8) subject to FF <= 5%.
Ties (within 0.5 pp of the best S): fewer flags, then smaller L (LC4 counts as 4), then smaller N. If no config meets FF <= 5%: STOP, write nothing but the tables (results/part2/dev_report.json,
results/part2/notes/p2_5_dev_grid_table.md) and exit with code 2 (the user's instruction of 2026-10-10 replaces the plan's "lowest FF" fallback: report, do not pick).
Nothing is chosen with held-out tasks 6 and 8; they are reported separately. The plan does not order a remaining tie (same flags, L and N; only the operating point differs):
fixed in deviations.md before any DEV episode: F1-max, recall>=0.99, fpr05, fpr03 in that order; the report says when it was needed.

--write: results/part2/notes/p2_5_dev_decision.{md,json}, configs/part2_test_arms.yaml (P0-P6 at the chosen config + P3-STAC, P3-teleport, P3-seed1, P3-seed2), results/part2/dev_report.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
HELD = (6, 8)
L_OF = {"L1": 1, "L2": 2, "L4": 4, "LC4": 4, "L6": 6, "L10": 10}
FF_LIMIT, TIE_PP = 0.05, 0.005
OP_ORDER = ("F1-max", "recall>=0.99", "fpr05", "fpr03")


def load(dev: Path, arm_names: list[str]) -> list[dict]:
    """Episodes where the detector never alarmed were run as P0 only (every arm is bitwise P0 until its first intervention): fill those arms in from P0."""
    eps = [json.load(gzip.open(f)) for f in sorted(dev.glob("ep_*.json.gz"))]
    for r in eps:
        p0 = r["arms"]["P0"]
        if len(r["arms"]) == 1:
            assert not any(d["alarm_a"] for d in p0["decisions_log"]), "episode with an accel alarm but only P0 logged"
            for nm in arm_names:
                r["arms"][nm] = dict(p0, events=[])
    return eps


def first_gate(arm: dict):
    for e in arm["events"]:
        if e["kind"] == "gate":
            return e
    return None


def metrics(eps: list[dict], name: str) -> dict:
    """S, SF, F, FF, FP, rescues, harms of one arm over `eps` (P0 is the counterfactual)."""
    n = len(eps)
    c = Counter()
    for r in eps:
        a, p0 = r["arms"][name], r["arms"]["P0"]
        succ, flag = bool(a["success"]), bool(a["flagged"])
        c["S"] += succ
        c["flag"] += flag
        c["SF"] += (not succ) and (not flag)
        c["flag_false"] += flag and bool(p0["success"])
        c["flag_true"] += flag and not p0["success"]
        c["rescue"] += succ and not p0["success"]
        c["harm"] += (not succ) and bool(p0["success"])
        c["rewinds"] += a["rewinds"]
        c["invalid"] += a["invalid_rewinds"]
        c["cut_short"] += bool(a["cut_short"])
        c["elapsed"] += a["elapsed"]
    return {"n": n, "S": c["S"] / n, "SF": c["SF"] / n, "F": c["flag"] / n, "FF": c["flag_false"] / n, "FP": (c["flag_true"] / c["flag"]) if c["flag"] else None,
            "flags": c["flag"], "flag_false": c["flag_false"], "flag_true": c["flag_true"], "successes": c["S"], "rescues": c["rescue"], "harms": c["harm"],
            "mean_rewinds": c["rewinds"] / n, "invalid_rewinds": c["invalid"], "cut_short": c["cut_short"], "mean_elapsed": c["elapsed"] / n}


def choose(rows: list[dict], fallback: bool = False) -> tuple[dict | None, dict]:
    """Apply the rule to rows = P3 configs with their seen-task metrics. Returns (winner, trace). `fallback`: results/part2/notes/p2_5_fallback_rule.txt (declared after DEV): no FF constraint."""
    feas = rows if fallback else [r for r in rows if r["seen"]["FF"] <= FF_LIMIT + 1e-12]
    if not feas:
        return None, {"feasible": 0, "lowest_FF": min(r["seen"]["FF"] for r in rows), "rule": "no config meets FF<=5%: stop and report (nothing chosen)"}
    smax = max(r["seen"]["S"] for r in feas)
    tied = [r for r in feas if r["seen"]["S"] >= smax - TIE_PP - 1e-12]
    key = lambda r: (r["seen"]["flags"], L_OF[r["target"]], r["N"])
    order = sorted(tied, key=key)
    top = [r for r in order if key(r) == key(order[0])]
    needed_last_resort = len(top) > 1
    win = sorted(top, key=lambda r: OP_ORDER.index(r["op"]))[0]
    return win, {"rule": "FALLBACK (no FF constraint; results/part2/notes/p2_5_fallback_rule.txt)" if fallback else "pre-registered", "feasible": len(feas), "best_S": smax, "tied_within_0.5pp": [r["name"] for r in tied], "after_flags_L_N": [r["name"] for r in top],
                 "last_resort_op_order_needed": needed_last_resort}


def first_alarm_truth(eps: list[dict], target: str, N: int) -> dict:
    """What had happened at the FIRST alarm of P0-failed episodes: oracle truth between the target and the settled frame (P2 at this target logs it before it acts)."""
    nm = f"P2-{target}-N{N}"
    rows = Counter()
    p2_unchanged = Counter()
    for r in eps:
        if r["arms"]["P0"]["success"]:
            continue
        g = first_gate(r["arms"][nm])
        if g is None:
            rows["no_alarm"] += 1
            continue
        key = "gray" if g["truth_gray"] else ("changed" if g["truth"] else "unchanged")
        rows[key] += 1
        if key == "unchanged":
            p2_unchanged["n"] += 1
            p2_unchanged["success"] += bool(r["arms"][nm]["success"])
            p2_unchanged["success_N1"] += bool(r["arms"][f"P2-{target}-N1"]["success"])
    out = dict(rows)
    out["alarmed"] = sum(v for k, v in rows.items() if k != "no_alarm")
    out["p2_retry_on_unchanged"] = {"n": p2_unchanged["n"], f"success_N{N}": p2_unchanged["success"], "success_N1": p2_unchanged["success_N1"]}
    return out


def harm_breakdown(eps: list[dict], name: str) -> dict:
    """P0-succeeded episodes that `name` lost: how."""
    out = Counter()
    rows = []
    for r in eps:
        a, p0 = r["arms"][name], r["arms"]["P0"]
        if not (p0["success"] and not a["success"]):
            continue
        gates = [e for e in a["events"] if e["kind"] == "gate"]
        g0 = gates[0] if gates else None
        if a["flagged"]:
            why = f"flagged:{a['flag_reason']}"
            if a["flag_reason"] == "changed" and g0:
                why += "|truth_changed" if gates[-1]["truth"] else "|truth_unchanged(gate_error)"
            elif a["flag_reason"] == "invalid_rewind":
                why += ""
        else:
            why = "silent_fail_after_retries" if a["rewinds"] else "silent_fail_no_rewind"
            if a["cut_short"]:
                why += "|cut_short"
        out[why] += 1
        rows.append({"task": r["task"], "init": r["init"], "why": why, "first_alarm_d": g0["d"] if g0 else None, "p0_decisions": p0["decisions"], "rewinds": a["rewinds"],
                     "elapsed": a["elapsed"], "truth_first_gate": (None if g0 is None else g0["truth"])})
    early = Counter()
    for x in rows:
        early["alarm_before_P0_finished"] += x["first_alarm_d"] is not None
    return {"n": len(rows), "how": dict(out), "episodes": rows}


def gate_vs_oracle(eps: list[dict], grid: list[dict], thr_of) -> dict:
    """All distinct (reference, settled) pairs scored in the loop vs the simulator truth, de-duplicated by content hash across arms, tagged with the arm's target. Accuracy etc. per
    (target, op) at the threshold that op uses for that target (`thr_of(op, target)`; the loop says CHANGED when s >= threshold)."""
    tgt_of = {g["name"]: g["target"] for g in grid}
    seen = {}
    for r in eps:
        for arm, a in r["arms"].items():
            pi = r.get("pair_index", {}).get(arm, [])
            for e in a["events"]:
                if e["kind"] != "gate" or e.get("pair") is None or e["pair"] >= len(pi):
                    continue
                key = (r["task"], r["init"], tgt_of[arm], tuple(pi[e["pair"]]))
                seen.setdefault(key, {"task": r["task"], "target": tgt_of[arm], "s": e["s"], "truth": e["truth"], "gray": e["truth_gray"], "gap": e["gap_steps"], "target_d": e["target_d"], "d": e["d"]})
    pairs = list(seen.values())

    def stats(ps, t):
        ng = [p for p in ps if not p["gray"]]
        pos = [p for p in ng if p["truth"]]
        neg = [p for p in ng if not p["truth"]]
        tp = sum(p["s"] >= t for p in pos)
        fp = sum(p["s"] >= t for p in neg)
        auc = None
        if pos and neg:
            sp, sn = np.array([p["s"] for p in pos]), np.array([p["s"] for p in neg])
            auc = float(((sp[:, None] > sn[None, :]).sum() + 0.5 * (sp[:, None] == sn[None, :]).sum()) / (len(sp) * len(sn)))
        return {"n": len(ps), "gray": len(ps) - len(ng), "changed": len(pos), "unchanged": len(neg), "recall": tp / len(pos) if pos else None, "misses": len(pos) - tp,
                "false_flag_rate": fp / len(neg) if neg else None, "false_flags": fp, "accuracy": ((tp + len(neg) - fp) / len(ng)) if ng else None, "auroc": auc}

    out = {"n_pairs": len(pairs), "by_target_op": {}}
    for tg in L_OF:
        for op in OP_ORDER:
            t = thr_of(op, tg)
            ps = [p for p in pairs if p["target"] == tg]
            o = {"threshold": t, "all": stats(ps, t), "seen": stats([p for p in ps if p["task"] not in HELD], t), "held_out": stats([p for p in ps if p["task"] in HELD], t), "by_gap_steps": {}, "by_task": {}}
            for g in sorted({p["gap"] for p in ps}):
                o["by_gap_steps"][str(g)] = stats([p for p in ps if p["gap"] == g], t)
            for k in sorted({p["task"] for p in ps}):
                o["by_task"][str(k)] = stats([p for p in ps if p["task"] == k], t)
            out["by_target_op"][f"{tg}/{op}"] = o
    return out


def test_arms(win: dict) -> dict:
    t, N, op = win["target"], win["N"], win["op"]
    arms = [{"name": "P0", "kind": "P0"}, {"name": "P1", "kind": "P1"}, {"name": "P2", "kind": "P2", "target": t, "N": N}, {"name": "P3", "kind": "P3", "target": t, "N": N, "op": op},
            {"name": "P4", "kind": "P4", "target": t, "N": N}, {"name": "P5", "kind": "P3", "trigger": "p5", "target": t, "N": N, "op": op}, {"name": "P6", "kind": "P6", "target": t, "op": op},
            {"name": "P3-STAC", "kind": "P3", "trigger": "stac", "target": t, "N": N, "op": op}, {"name": "P3-teleport", "kind": "P3", "target": t, "N": N, "op": op, "teleport": True},
            {"name": "P3-seed1", "kind": "P3", "target": t, "N": N, "op": op, "seed": 1}, {"name": "P3-seed2", "kind": "P3", "target": t, "N": N, "op": op, "seed": 2}]
    return {"arms": {"test": arms}}


def fmt(x, pct=True):
    return "-" if x is None else (f"{100 * x:.1f}" if pct else f"{x:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(ROOT / "results" / "part2" / "dev"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="the fallback rule declared after DEV (results/part2/notes/p2_5_fallback_rule.txt): drop the FF constraint")
    a = ap.parse_args()
    grid = yaml.safe_load(open(ROOT / "configs" / "part2_dev_grid.yaml"))["arms"]["dev"]
    eps = load(Path(a.dev), [x["name"] for x in grid])
    seen = [r for r in eps if r["task"] not in HELD]
    held = [r for r in eps if r["task"] in HELD]
    names = set(eps[0]["arms"])
    assert all(set(r["arms"]) == names for r in eps), "episodes with different arm sets"
    assert {s["name"] for s in grid} | {"P0"} == names, "arms in the logs differ from configs/part2_dev_grid.yaml"
    print(f"{len(eps)} DEV episodes ({len(seen)} seen, {len(held)} held-out); {len(grid)} grid arms")
    p3 = [s for s in grid if s["kind"] == "P3"]
    rows = []
    for s in p3:
        rows.append({"name": s["name"], "target": s["target"], "N": s["N"], "op": s["op"], "seen": metrics(seen, s["name"]), "held_out": metrics(held, s["name"]) if held else None, "all": metrics(eps, s["name"])})
    win, trace = choose(rows, a.fallback)
    frozen = yaml.safe_load(open(ROOT / "configs" / "part2_frozen.yaml"))["deltadino"]["thresholds_val_label_v2"][0]
    gcal = yaml.safe_load(open(ROOT / "configs" / "part2_gate_cal.yaml"))["thresholds"][0]
    thr_of = lambda op, tg: float(gcal[op][tg]) if op in gcal else float(frozen[op])
    ranked = sorted(rows, key=lambda r: (-r["seen"]["S"], r["seen"]["FF"]))
    if win is None:
        print("NO P3 CONFIG MEETS FF <= 5% on the DEV seen tasks: stopping, nothing chosen.", trace)
        print("lowest-FF configs and the top 5 by S:")
        for r in sorted(rows, key=lambda r: (r["seen"]["FF"], -r["seen"]["S"]))[:5] + ranked[:5]:
            x = r["seen"]
            print(f"  {r['name']:16s} S {fmt(x['S'])} SF {fmt(x['SF'])} F {fmt(x['F'])} FF {fmt(x['FF'])} flags {x['flags']} rescues {x['rescues']} harms {x['harms']}")
        if a.write:
            (ROOT / "results" / "part2").mkdir(parents=True, exist_ok=True)
            (ROOT / "results" / "part2" / "dev_report.json").write_text(json.dumps({"grid": rows, "trace": trace, "p0": metrics(seen, "P0")}, indent=1, default=float))
            tl = ["# P2.5 DEV grid (seen tasks): NO config meets FF <= 5%; nothing chosen", "", "| config | S | SF | F | FF | flags | rescues | harms |", "|---|---|---|---|---|---|---|---|"]
            for r in ranked:
                x = r["seen"]
                tl.append(f"| {r['name']} | {fmt(x['S'])} | {fmt(x['SF'])} | {fmt(x['F'])} | {fmt(x['FF'])} | {x['flags']} | {x['rescues']} | {x['harms']} |")
            (ROOT / "results" / "part2" / "notes" / "p2_5_dev_grid_table.md").write_text("\n".join(tl) + "\n")
        sys.exit(2)
    print("chosen:", win["name"], trace)
    # ---- tables
    p0 = {"seen": metrics(seen, "P0"), "held_out": metrics(held, "P0") if held else None, "all": metrics(eps, "P0")}
    t, N, op = win["target"], win["N"], win["op"]
    chosen_arms = {"P0": p0, "P2": {k: metrics(v, f"P2-{t}-N{N}") for k, v in (("seen", seen), ("held_out", held), ("all", eps)) if v},
                   "P3": {"seen": win["seen"], "held_out": win["held_out"], "all": win["all"]}, "P4": {k: metrics(v, f"P4-{t}-N{N}") for k, v in (("seen", seen), ("held_out", held), ("all", eps)) if v}}
    report = {
        "n_episodes": len(eps), "n_seen": len(seen), "n_held_out": len(held),
        "grid": rows, "choice": {"name": win["name"], "target": t, "N": N, "op": op, "trace": trace},
        "chosen_arms": chosen_arms,
        "first_alarm_truth": {"seen": first_alarm_truth(seen, t, N), "held_out": first_alarm_truth(held, t, N), "all": first_alarm_truth(eps, t, N)},
        "first_alarm_truth_by_target": {tt: {"seen": first_alarm_truth(seen, tt, 3), "all": first_alarm_truth(eps, tt, 3)} for tt in L_OF},
        "harms_P3": {"seen": harm_breakdown(seen, win["name"]), "held_out": harm_breakdown(held, win["name"]) if held else None},
        "p2_p4_per_target": {f"{k}-{tt}-N{n}": {"seen": metrics(seen, f"{k}-{tt}-N{n}")} for k in ("P2", "P4") for tt in L_OF for n in (1, 3)},
        "gate_vs_oracle": gate_vs_oracle(eps, grid, thr_of),
        "secs": {"total": sum(r["secs"] for r in eps), "mean": float(np.mean([r["secs"] for r in eps])), "max": max(r["secs"] for r in eps)},
        "git_episodes": sorted({r["git"] for r in eps}),
    }
    print("top 5 by seen S:")
    for r in ranked[:5]:
        s = r["seen"]
        print(f"  {r['name']:16s} S {fmt(s['S'])} SF {fmt(s['SF'])} F {fmt(s['F'])} FF {fmt(s['FF'])} flags {s['flags']} rescues {s['rescues']} harms {s['harms']}")
    if not a.write:
        return
    # ---- write
    (ROOT / "results" / "part2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "part2" / "dev_report.json").write_text(json.dumps(report, indent=1, default=float))
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dec = {"rule_used": "FALLBACK rule (no FF constraint): pre-registered rule infeasible on DEV, fallback declared after DEV before TEST (results/part2/notes/p2_5_fallback_rule.txt)" if a.fallback else "pre-registered",
           "chosen": {"target": t, "N": N, "op": op, "arm": win["name"]}, "rule": trace, "seen_metrics": win["seen"], "held_out_metrics": win["held_out"], "n_dev_episodes": len(eps),
           "n_seen": len(seen), "code_git": git, "dev_episode_git": report["git_episodes"], "detector": "accel", "deltadino": "model C (static mask only), seed 0, op " + op + ", threshold(s) " + json.dumps({tg: thr_of(op, tg) for tg in L_OF}),
           "grid_arms": len(grid), "feasible_configs": trace.get("feasible")}
    (ROOT / "results" / "part2" / "notes" / "p2_5_dev_decision.json").write_text(json.dumps(dec, indent=1, default=float))
    yaml.safe_dump(test_arms(win), open(ROOT / "configs" / "part2_test_arms.yaml", "w"), sort_keys=False, default_flow_style=None)
    lines = ["# P2.5: DEV decision", "",
             ("**FALLBACK RULE** (`p2_5_fallback_rule.txt`, committed before applying; declared after DEV because the pre-registered rule was infeasible: no config with FF <= 5%, lowest 5.6%): maximise S on the DEV **seen** tasks, no FF constraint; ties within 0.5 pp: fewer flags, then smaller L, then smaller N, then op order F1-max, recall>=0.99, fpr05, fpr03. TEST is confirmatory for H1-H3, not blind to DEV."
              if a.fallback else "Rule (pre-registered, `p2_4_analysis_plan.md`): maximise S on the DEV **seen** tasks subject to FF <= 5%; ties within 0.5 pp: fewer flags, then smaller L, then smaller N; if no config meets FF <= 5% the run stops and reports (no choice)."),
             f"Data: {len(eps)} DEV episodes (10 tasks x init 0-19 x `p2dev-1`); {len(seen)} on the 8 seen tasks decide, {len(held)} on held-out tasks 6 and 8 are reported only. Detector: accel; DeltaDINO model C seed 0. DEV numbers choose a config; they are not claims.", "",
             f"## Choice: **{win['name']}** (target `{t}`, N = {N}, operating point `{op}`)", "", f"Rule trace: `{json.dumps(trace)}`", "",
             "| | S | SF | F | FF | FP | flags | rescues | harms |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, d in chosen_arms.items():
        s = d["seen"]
        lines.append(f"| {nm} (seen, n={s['n']}) | {fmt(s['S'])} | {fmt(s['SF'])} | {fmt(s['F'])} | {fmt(s['FF'])} | {fmt(s['FP'])} | {s['flags']} | {s['rescues']} | {s['harms']} |")
    lines += ["", "## Full P3 grid (seen tasks; percent of episodes)", "", "| config | S | SF | F | FF | FP | flags | rescues | harms | held-out S | held-out FF | meets FF<=5% |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ranked:
        s, h = r["seen"], r["held_out"] or {}
        lines.append(f"| {r['name']} | {fmt(s['S'])} | {fmt(s['SF'])} | {fmt(s['F'])} | {fmt(s['FF'])} | {fmt(s['FP'])} | {s['flags']} | {s['rescues']} | {s['harms']} | {fmt(h.get('S'))} | {fmt(h.get('FF'))} | {'yes' if s['FF'] <= FF_LIMIT + 1e-12 else 'no'} |")
    lines += ["", "## P2 and P4 per target (seen tasks)", "", "| arm | S | SF | F | FF | flags | rescues | harms |", "|---|---|---|---|---|---|---|---|"]
    for k, v in report["p2_p4_per_target"].items():
        s = v["seen"]
        lines.append(f"| {k} | {fmt(s['S'])} | {fmt(s['SF'])} | {fmt(s['F'])} | {fmt(s['FF'])} | {s['flags']} | {s['rescues']} | {s['harms']} |")
    lines += ["", f"P0 (seen): S {fmt(p0['seen']['S'])}, SF {fmt(p0['seen']['SF'])}.", "",
              "Files: `results/part2/dev_report.json` (all tables), `results/part2/notes/p2_5_dev_decision.json` (machine-readable, unlocks TEST), `configs/part2_test_arms.yaml`.", ""]
    (ROOT / "results" / "part2" / "notes" / "p2_5_dev_decision.md").write_text("\n".join(lines))
    print("written")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
