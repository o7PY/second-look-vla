"""P2.7: analysis of a TEST run, exactly as pre-registered (results/part2/notes/p2_4_analysis_plan.md + SPEC_part2 P2.7).

  python scripts/p2_analyse.py [--dir results/part2/test] [--out results/part2/results.json] [--md results/part2/notes/p2_7_results.md] [--boot 10000]

* unit = one episode (task, init, seed); cluster = (task, init) (the 3 noise seeds of an init state are one cluster); CIs = cluster bootstrap, B resamples, percentile 95% (one-sided 97.5% bound
  for the non-inferiority part of H2); paired binary comparisons also get an exact McNemar test.
* Arms not present in an episode's log did not run because their trigger never fired: they equal P0 there (every arm is bitwise P0 until its first intervention).
* H1: S(P3)-S(P0) > 0. H2: SF(P3) < SF(P2) AND S(P3) >= S(P2) - 1 pp (the lower bound of the one-sided 97.5% CI of the difference, i.e. the 2.5th percentile, must be >= -1 pp; H2 passes only if both).
  H3: S(P3)-S(P5) > 0. One-sided p = share of resamples with the difference <= 0 (>= 0 for SF(P3) < SF(P2); <= -1 pp for the non-inferiority part), floored at 1/(B+1);
  H2's p is the larger of its two. Holm over (H1, H2, H3); a hypothesis is "supported" when its Holm-adjusted p < 0.05 (H2 also needs the non-inferiority bound).
* Main numbers on the 8 SEEN tasks (all but 6 and 8); held-out (6, 8) and all 10 tasks are reported separately; the init 20-39 ("clean": 40-49 overlap CAL) subset too.
Writes results.json and a markdown report.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
HELD = (6, 8)
MAIN = ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
SECOND = ["P3-STAC", "P3-teleport", "P3-seed1", "P3-seed2"]
ARMS = MAIN + SECOND


def load(d: Path):
    eps = []
    for f in sorted(d.glob("ep_*.json.gz")):
        r = json.load(gzip.open(f))
        p0 = r["arms"]["P0"]
        arms = {}
        for nm in ARMS:
            a = r["arms"].get(nm)
            arms[nm] = a if a is not None else dict(p0, events=[], rewinds=0, invalid_rewinds=0, cut_short=False, flagged=False, flag_reason=None)
        eps.append({"task": r["task"], "init": r["init"], "seed_id": r["seed_id"], "arms": arms, "pair_index": r.get("pair_index", {}), "file": f.name,
                    "arms_present": [k for k in r["arms"] if k != "P0"], "secs": r["secs"], "git": r["git"]})
    return eps


def per_episode(eps, arm):
    """arrays aligned with eps: success, flag, silent, flag_false, flag_true, rescue, harm, rewinds, elapsed, cut_short, invalid"""
    cols = defaultdict(list)
    for e in eps:
        a, p0 = e["arms"][arm], e["arms"]["P0"]
        s, fl = bool(a["success"]), bool(a["flagged"])
        cols["S"].append(s)
        cols["F"].append(fl)
        cols["SF"].append((not s) and not fl)
        cols["FF"].append(fl and bool(p0["success"]))
        cols["flag_true"].append(fl and not p0["success"])
        cols["rescue"].append(s and not p0["success"])
        cols["harm"].append((not s) and bool(p0["success"]))
        cols["rewinds"].append(a["rewinds"])
        cols["elapsed"].append(a["elapsed"])
        cols["cut_short"].append(bool(a["cut_short"]))
        cols["invalid"].append(a["invalid_rewinds"])
    return {k: np.array(v, float) for k, v in cols.items()}


class Boot:
    def __init__(self, eps, B: int, seed: int = 0):
        keys = sorted({(e["task"], e["init"]) for e in eps})
        self.idx = {k: i for i, k in enumerate(keys)}
        self.cl = np.array([self.idx[(e["task"], e["init"])] for e in eps])
        self.C, self.B = len(keys), B
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.C, np.ones(self.C) / self.C, size=B).astype(float)        # (B, C) cluster counts

    def sums(self, x):
        return np.bincount(self.cl, weights=x, minlength=self.C)

    def rate(self, num, den=None):
        n = self.sums(np.asarray(num, float))
        d = self.sums(np.ones(len(self.cl)) if den is None else np.asarray(den, float))
        est = n.sum() / d.sum() if d.sum() else None
        bs = (self.W @ n) / np.maximum(self.W @ d, 1e-12)
        ok = (self.W @ d) > 0
        bs = bs[ok]
        return est, ([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else None), bs

    def diff(self, x, y):
        """mean(x) - mean(y) with bootstrap samples (x, y per episode arrays)."""
        n = self.sums(np.asarray(x, float) - np.asarray(y, float))
        d = self.sums(np.ones(len(self.cl)))
        est = n.sum() / d.sum()
        bs = (self.W @ n) / (self.W @ d)
        return est, bs


def mcnemar(a_succ, b_succ):
    """Exact two-sided McNemar on paired binary outcomes: b = a only, c = b only."""
    a, b = np.asarray(a_succ, bool), np.asarray(b_succ, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return {"a_only": n10, "b_only": n01, "p": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def table(eps, B: int, arms=ARMS) -> dict:
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
        row["mean_retries"] = float(c["rewinds"].mean())
        row["mean_elapsed"] = float(c["elapsed"].mean())
        row["cut_short"] = int(c["cut_short"].sum())
        row["invalid_rewinds"] = int(c["invalid"].sum())
        out["arms"][arm] = row
    return out


def hypotheses(eps, B: int) -> dict:
    bt = Boot(eps, B)
    f = lambda arm: per_episode(eps, arm)
    p0, p2, p3, p5 = f("P0"), f("P2"), f("P3"), f("P5")
    floor = 1.0 / (B + 1)
    h = {}
    est, bs = bt.diff(p3["S"], p0["S"])
    h["H1"] = {"claim": "S(P3) - S(P0) > 0", "diff": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(max(np.mean(bs <= 0), floor)), "mcnemar": mcnemar(p3["S"], p0["S"])}
    est_sf, bs_sf = bt.diff(p3["SF"], p2["SF"])
    est_s, bs_s = bt.diff(p3["S"], p2["S"])
    pa, pb = float(max(np.mean(bs_sf >= 0), floor)), float(max(np.mean(bs_s <= -0.01), floor))
    lower = float(np.percentile(bs_s, 2.5))
    h["H2"] = {"claim": "SF(P3) < SF(P2) AND S(P3) >= S(P2) - 1 pp", "SF_diff": est_sf, "SF_ci95": [float(np.percentile(bs_sf, 2.5)), float(np.percentile(bs_sf, 97.5))], "p_a": pa,
               "S_diff": est_s, "S_ci95": [lower, float(np.percentile(bs_s, 97.5))], "S_lower_bound_one_sided_97.5": lower, "noninferior": bool(lower >= -0.01), "p_b": pb, "p": max(pa, pb),
               "mcnemar_S": mcnemar(p3["S"], p2["S"]), "mcnemar_SF": mcnemar(p3["SF"], p2["SF"])}
    est, bs = bt.diff(p3["S"], p5["S"])
    h["H3"] = {"claim": "S(P3) - S(P5) > 0", "diff": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(max(np.mean(bs <= 0), floor)), "mcnemar": mcnemar(p3["S"], p5["S"])}
    order = sorted(["H1", "H2", "H3"], key=lambda k: h[k]["p"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (3 - i) * h[k]["p"]))
        h[k]["p_holm"] = run
        h[k]["supported"] = bool(run < 0.05 and (h[k]["noninferior"] if k == "H2" else True))
    # descriptive differences with CIs
    desc = {}
    for nm, (a, b) in {"P4-P3 (cost of gate errors) S": ("P4", "P3"), "P6-P3 (value of rewinding) S": ("P6", "P3"), "P1-P0 (old null) S": ("P1", "P0"), "P2-P0 S": ("P2", "P0"),
                       "P3-teleport - P3 S": ("P3-teleport", "P3"), "P3-STAC - P3 S": ("P3-STAC", "P3"), "P3-seed1 - P3 S": ("P3-seed1", "P3"), "P3-seed2 - P3 S": ("P3-seed2", "P3")}.items():
        est, bs = bt.diff(f(a)["S"], f(b)["S"])
        desc[nm] = {"diff": est, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "mcnemar": mcnemar(f(a)["S"], f(b)["S"])}
    h["descriptive"] = desc
    return h


def first_alarm_table(eps) -> dict:
    """P3's first second look: simulator truth x gate verdict x outcome (and what P0 did)."""
    rows = Counter()
    for e in eps:
        a = e["arms"]["P3"]
        g = [x for x in a["events"] if x["kind"] == "gate"]
        if not g:
            rows[("no alarm", "-", "success" if a["success"] else "silent_fail", "P0 " + ("success" if e["arms"]["P0"]["success"] else "fail"))] += 1
            continue
        g0 = g[0]
        truth = "gray" if g0["truth_gray"] else ("changed" if g0["truth"] else "unchanged")
        out = "success" if a["success"] else ("flagged" if a["flagged"] else "silent_fail")
        rows[(truth, g0["verdict"], out, "P0 " + ("success" if e["arms"]["P0"]["success"] else "fail"))] += 1
    return {"|".join(k): v for k, v in sorted(rows.items())}


def gate_stats(eps, thr_main: float) -> dict:
    """All distinct gate pairs of seed-0 arms vs the simulator truth. Dedup by content hash of (reference, settled) within an episode."""
    seen = {}
    for e in eps:
        for arm, a in e["arms"].items():
            if arm in ("P3-seed1", "P3-seed2"):
                continue
            pi = e["pair_index"].get(arm, [])
            for ev in a["events"]:
                if ev["kind"] != "gate" or ev.get("pair") is None or ev["pair"] >= len(pi):
                    continue
                key = (e["task"], e["init"], e["seed_id"], tuple(pi[ev["pair"]]))
                seen.setdefault(key, {"task": e["task"], "s": ev["s"], "truth": ev["truth"], "gray": ev["truth_gray"], "gap": ev["gap_steps"], "first": ev["rewinds"] == 0})
    pairs = list(seen.values())

    def st(ps, t):
        ng = [p for p in ps if not p["gray"]]
        pos = np.array([p["s"] for p in ng if p["truth"]])
        neg = np.array([p["s"] for p in ng if not p["truth"]])
        auc = None
        if len(pos) and len(neg):
            auc = float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
        return {"n": len(ps), "gray": len(ps) - len(ng), "changed": int(len(pos)), "unchanged": int(len(neg)), "auroc": auc, "recall": float((pos >= t).mean()) if len(pos) else None,
                "fpr": float((neg >= t).mean()) if len(neg) else None, "misses": int((pos < t).sum()), "false_flags": int((neg >= t).sum())}
    out = {"n_pairs": len(pairs), "threshold": thr_main}
    for scope, f in (("all", lambda p: True), ("seen", lambda p: p["task"] not in HELD), ("held_out", lambda p: p["task"] in HELD), ("first_gate_only/seen", lambda p: p["first"] and p["task"] not in HELD)):
        out[scope] = st([p for p in pairs if f(p)], thr_main)
    out["by_gap_steps"] = {str(g): st([p for p in pairs if p["gap"] == g], thr_main) for g in sorted({p["gap"] for p in pairs})}
    out["by_task"] = {str(k): st([p for p in pairs if p["task"] == k], thr_main) for k in sorted({p["task"] for p in pairs})}
    return out


def timing(eps) -> dict:
    out = {}
    for arm in ("P0", "P2", "P3", "P4", "P6"):
        el = np.array([e["arms"][arm]["elapsed"] for e in eps], float)
        out[arm] = {"mean": float(el.mean()), "median": float(np.median(el)), "q10_q90": [float(np.percentile(el, 10)), float(np.percentile(el, 90))], "hit_budget_300": int((el >= 300).sum()),
                    "cut_short": int(sum(bool(e["arms"][arm]["cut_short"]) for e in eps))}
    return out


def invalid(eps) -> dict:
    out = {}
    for arm in ("P2", "P3", "P4"):
        rows = [e for e in eps if e["arms"][arm]["invalid_rewinds"]]
        out[arm] = {"episodes_with_invalid_rewind": len(rows), "outcomes": dict(Counter(("success" if e["arms"][arm]["success"] else ("flagged:" + str(e["arms"][arm]["flag_reason"]))) + (" / P0 success" if e["arms"]["P0"]["success"] else " / P0 fail") for e in rows))}
    return out


def per_task(eps) -> dict:
    out = {}
    for t in range(10):
        es = [e for e in eps if e["task"] == t]
        if not es:
            continue
        out[str(t)] = {"n": len(es), "held_out": t in HELD, **{arm: {"S": float(np.mean([bool(e["arms"][arm]["success"]) for e in es])), "rescue": int(sum(e["arms"][arm]["success"] and not e["arms"]["P0"]["success"] for e in es)),
                                                                     "harm": int(sum((not e["arms"][arm]["success"]) and e["arms"]["P0"]["success"] for e in es)), "flags": int(sum(e["arms"][arm]["flagged"] for e in es))}
                                                       for arm in ("P0", "P2", "P3", "P4")}}
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def ci_s(c):
    return "" if not c else f" [{pct(c[0])}, {pct(c[1])}]"


def md_table(title, tb):
    if not tb:
        return [f"### {title}", "", "(no episodes)", ""]
    L = [f"### {title} ({tb['n_episodes']} episodes, {tb['n_clusters']} clusters)", "", "| arm | S % [95% CI] | SF % | F % | FF % | FP % | rescues | harms | mean retries | mean elapsed | cut short | invalid |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in tb["arms"].items():
        L.append(f"| {arm} | {pct(r['S']['est'])}{ci_s(r['S']['ci'])} | {pct(r['SF']['est'])}{ci_s(r['SF']['ci'])} | {pct(r['F']['est'])} | {pct(r['FF']['est'])}{ci_s(r['FF']['ci'])} | {pct(r['FP']['est'])} ({r['FP']['flag_true']}/{r['FP']['flags']}) | "
                 f"{r['rescue']['count']} | {r['harm']['count']} | {r['mean_retries']:.2f} | {r['mean_elapsed']:.0f} | {r['cut_short']} | {r['invalid_rewinds']} |")
    return L + [""]


def md_hyp(title, h):
    L = [f"### {title}", "", "| hypothesis | estimate [95% CI] | p (one-sided) | Holm p | supported |", "|---|---|---|---|---|"]
    L.append(f"| H1 S(P3)-S(P0) > 0 | {pct(h['H1']['diff'])} pp{ci_s([100 * x / 100 for x in h['H1']['ci95']])} | {h['H1']['p']:.4f} | {h['H1']['p_holm']:.4f} | {h['H1']['supported']} |")
    H2 = h["H2"]
    L.append(f"| H2 SF(P3)-SF(P2) < 0 AND S(P3)-S(P2) >= -1 pp | SF diff {pct(H2['SF_diff'])} pp{ci_s(H2['SF_ci95'])}; S diff {pct(H2['S_diff'])} pp{ci_s(H2['S_ci95'])} (lower bound {pct(H2['S_lower_bound_one_sided_97.5'])}, noninferior {H2['noninferior']}) | a {H2['p_a']:.4f}, b {H2['p_b']:.4f} -> {H2['p']:.4f} | {H2['p_holm']:.4f} | {H2['supported']} |")
    L.append(f"| H3 S(P3)-S(P5) > 0 | {pct(h['H3']['diff'])} pp{ci_s(h['H3']['ci95'])} | {h['H3']['p']:.4f} | {h['H3']['p_holm']:.4f} | {h['H3']['supported']} |")
    L += ["", "Descriptive differences (S, percentage points, 95% CI; exact McNemar p):", ""]
    for k, v in h["descriptive"].items():
        L.append(f"- {k}: {pct(v['diff'])}{ci_s(v['ci95'])}; McNemar a-only {v['mcnemar']['a_only']}, b-only {v['mcnemar']['b_only']}, p = {v['mcnemar']['p']:.3f}")
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "part2" / "test"))
    ap.add_argument("--out", default=str(ROOT / "results" / "part2" / "results.json"))
    ap.add_argument("--md", default=str(ROOT / "results" / "part2" / "notes" / "p2_7_results.md"))
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    eps = load(Path(a.dir))
    seen = [e for e in eps if e["task"] not in HELD]
    held = [e for e in eps if e["task"] in HELD]
    clean = [e for e in seen if e["init"] < 40]
    ovl = [e for e in seen if e["init"] >= 40]
    thr_main = yaml.safe_load(open(ROOT / "configs" / "part2_gate_cal.yaml"))["thresholds"][0]["fpr05"]["L6"]
    res = {"n_episodes": len(eps), "n_seen": len(seen), "n_held_out": len(held), "git_episodes": sorted({e["git"] for e in eps}), "bootstrap": a.boot,
           "tables": {"seen": table(seen, a.boot), "held_out": table(held, a.boot), "all": table(eps, a.boot), "seen_init20_39": table(clean, a.boot), "seen_init40_49": table(ovl, a.boot)},
           "hypotheses": {"seen": hypotheses(seen, a.boot), "seen_init20_39": hypotheses(clean, a.boot) if clean else None, "all": hypotheses(eps, a.boot)},
           "first_alarm_truth_gate_outcome": {"seen": first_alarm_table(seen), "held_out": first_alarm_table(held)},
           "gate_in_loop": gate_stats(eps, thr_main), "per_task": per_task(eps), "timing": {"seen": timing(seen), "all": timing(eps)}, "invalid_rewinds": invalid(eps)}
    # first-alarm truth on failed episodes (the 'could a rewind help at all' table)
    ft = {}
    for scope, es in (("seen", seen), ("held_out", held)):
        c = Counter()
        for e in es:
            if e["arms"]["P0"]["success"]:
                continue
            g = [x for x in e["arms"]["P2"]["events"] if x["kind"] == "gate"]
            c["no alarm" if not g else ("gray" if g[0]["truth_gray"] else ("changed" if g[0]["truth"] else "unchanged"))] += 1
        ft[scope] = dict(c)
    res["first_alarm_truth_failed_P0"] = ft
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    L = ["# P2.7 results (TEST, chosen config P3-L6-N3-fpr05; analysis exactly as pre-registered)", "", f"{len(eps)} episodes ({len(seen)} seen, {len(held)} held-out), code commits {', '.join(res['git_episodes'])}, bootstrap B = {a.boot}.", ""]
    L += ["## 1. Main table (8 seen tasks)", ""] + md_table("Seen tasks", res["tables"]["seen"])
    L += ["## 2. H1-H3 (seen tasks, pre-registered family, Holm)", ""] + md_hyp("All TEST init states 20-49", res["hypotheses"]["seen"])
    if res["hypotheses"]["seen_init20_39"]:
        L += md_hyp("Clean init states 20-39 only (40-49 overlap CAL)", res["hypotheses"]["seen_init20_39"])
    L += md_table("Seen, init 20-39 (clean)", res["tables"]["seen_init20_39"]) + md_table("Seen, init 40-49 (overlap CAL)", res["tables"]["seen_init40_49"])
    L += ["## 3. Held-out tasks (6, 8) and all 10 tasks", ""] + md_table("Held-out", res["tables"]["held_out"]) + md_table("All 10 tasks", res["tables"]["all"]) + md_hyp("All 10 tasks (descriptive; H-family was pre-registered on seen tasks)", res["hypotheses"]["all"])
    L += ["## 4. First alarm: simulator truth x gate verdict x outcome (P3; counts of episodes)", ""]
    for scope in ("seen", "held_out"):
        L += [f"**{scope}**", "", "| truth | gate | P3 outcome | P0 | n |", "|---|---|---|---|---|"]
        for k, v in res["first_alarm_truth_gate_outcome"][scope].items():
            L.append("| " + " | ".join(k.split("|")) + f" | {v} |")
        L.append("")
    L += [f"Scene truth at the first alarm of failed-P0 episodes (blind P2 gate log): seen {res['first_alarm_truth_failed_P0']['seen']}, held-out {res['first_alarm_truth_failed_P0']['held_out']}.", ""]
    g = res["gate_in_loop"]
    L += ["## 5. DeltaDINO in the loop vs simulator truth (all distinct seed-0 gate pairs; threshold fpr05 for L6 = %+.3f, gate says CHANGED when s >= threshold)" % thr_main, "", "| scope | pairs | changed / unchanged / gray | AUROC | recall | FPR | misses | false flags |", "|---|---|---|---|---|---|---|---|"]
    for k in ("all", "seen", "held_out", "first_gate_only/seen"):
        x = g[k]
        L.append(f"| {k} | {x['n']} | {x['changed']} / {x['unchanged']} / {x['gray']} | {x['auroc'] if x['auroc'] is None else round(x['auroc'], 3)} | {pct(x['recall'])} | {pct(x['fpr'])} | {x['misses']} | {x['false_flags']} |")
    L += ["", "By gap (steps): " + "; ".join(f"{k}: n {v['n']}, AUROC {v['auroc'] if v['auroc'] is None else round(v['auroc'], 3)}, recall {pct(v['recall'])}, FPR {pct(v['fpr'])}" for k, v in g["by_gap_steps"].items()), "",
          "By task: " + "; ".join(f"{k}: n {v['n']}, AUROC {v['auroc'] if v['auroc'] is None else round(v['auroc'], 3)}, recall {pct(v['recall'])}, FPR {pct(v['fpr'])}" for k, v in g["by_task"].items()), "",
          "Sealed reference (Part 1, C): AUROC 0.993, recall 0.921, 1.2% flagged; CAL in-loop (L6, fpr05): FPR 4.2%, recall 89.5%.", ""]
    L += ["## 6. Per task (S %, rescues, harms, flags)", "", "| task | n | P0 S | P2 S (resc/harm) | P3 S (resc/harm/flags) | P4 S (resc/harm/flags) |", "|---|---|---|---|---|---|"]
    for t, r in res["per_task"].items():
        L.append(f"| {t}{' (held-out)' if r['held_out'] else ''} | {r['n']} | {pct(r['P0']['S'])} | {pct(r['P2']['S'])} ({r['P2']['rescue']}/{r['P2']['harm']}) | {pct(r['P3']['S'])} ({r['P3']['rescue']}/{r['P3']['harm']}/{r['P3']['flags']}) | {pct(r['P4']['S'])} ({r['P4']['rescue']}/{r['P4']['harm']}/{r['P4']['flags']}) |")
    L += ["", "## 7. Time (elapsed steps; budget 300)", "", "| arm | mean | median | 10-90% | at budget | retries cut short |", "|---|---|---|---|---|---|"]
    for arm, x in res["timing"]["seen"].items():
        L.append(f"| {arm} | {x['mean']:.0f} | {x['median']:.0f} | {x['q10_q90'][0]:.0f}-{x['q10_q90'][1]:.0f} | {x['hit_budget_300']} | {x['cut_short']} |")
    L += ["", "## 8. Invalid rewinds (penetration > 1 cm), all episodes", ""] + [f"- {arm}: {v['episodes_with_invalid_rewind']} episodes; outcomes {v['outcomes']}" for arm, v in res["invalid_rewinds"].items()]
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
