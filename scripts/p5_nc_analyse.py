"""P5.0 analysis, exactly as pre-registered (results/part5/notes/p5_0_plan.txt).
  python scripts/p5_nc_analyse.py [--dir results/part5/nc/states] [--boot 10000]  -> results/part5/nc/results.json, results/part5/notes/p5_0_results.md
Primary: population A, mode R: GAP = mean any-of-64 - mean p_s (any-of-K if K < 64), cluster bootstrap over (task, init), B = 10,000, verdict on the point estimate (GO >= 20 pp, WEAK 10-20, NO-GO < 10)."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C
from p2_analyse import Boot

ROOT = C.ROOT
KS = [1, 2, 4, 8, 16, 32, 64]


def anyk(n: int, K: int, k: int) -> float:
    return 1.0 - comb(K - n, k) / comb(K, k)


def load_states(d: Path) -> list[dict]:
    return [json.load(gzip.open(f)) for f in sorted(d.glob("s_*.json.gz"))]


def per_state(rec: dict, mode: str):
    b = [v for k, v in rec["branches"].items() if k.startswith(mode)]
    if not b:
        return None
    n = int(sum(x["success"] for x in b))
    return {"n": n, "K": len(b), "p": n / len(b), "alarm_again_rescued": [bool(x["alarm_again"]) for x in b if x["success"]], "alarm_again_all": [bool(x["alarm_again"]) for x in b]}


def curve(rows: list[dict], B: Boot) -> dict:
    out = {}
    K = min(r["K"] for r in rows)
    for k in [k for k in KS if k <= K]:
        v = np.array([anyk(r["n"], r["K"], k) for r in rows])
        est, bs = B.diff(v, np.zeros(len(v)))
        out[k] = {"est": est, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def mean_ci(v, B: Boot):
    est, bs = B.diff(np.asarray(v, float), np.zeros(len(v)))
    return {"est": est, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def pop_stats(recs: list[dict], mode: str, B_n: int) -> dict:
    rows, keep = [], []
    for r in recs:
        s = per_state(r, mode)
        if s is not None:
            rows.append(s); keep.append(r)
    if not rows:
        return {"n_states": 0}
    eps = [{"task": r["state"]["task"], "init": r["state"]["init"]} for r in keep]
    B = Boot(eps, B_n)
    K = min(r["K"] for r in rows)
    kp = max(k for k in KS if k <= K)
    anyp = np.array([anyk(r["n"], r["K"], kp) for r in rows]); p = np.array([r["p"] for r in rows])
    est, bs = B.diff(anyp, p)
    out = {"n_states": len(rows), "n_clusters": B.C, "K": K, "k_primary": kp, "mean_p": mean_ci(p, B), "any_k": mean_ci(anyp, B), "gap": {"est": est, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
           "curve": curve(rows, B), "dist": {"zero": float(np.mean([r["n"] == 0 for r in rows])), "mid": float(np.mean([0 < r["p"] < 0.5 for r in rows])), "high": float(np.mean([r["p"] >= 0.5 for r in rows]))},
           "alarm_again_rescued": float(np.mean([a for r in rows for a in r["alarm_again_rescued"]])) if any(r["alarm_again_rescued"] for r in rows) else None,
           "alarm_again_all": float(np.mean([a for r in rows for a in r["alarm_again_all"]]))}
    out["_rows"] = [{"task": r_["state"]["task"], "init": r_["state"]["init"], "scene": r_["state"]["scene"], "p": r["p"], "n": r["n"], "K": r["K"], "anyk": float(a)} for r_, r, a in zip(keep, rows, anyp)]
    return out


def breakdown(rows: list[dict], key) -> dict:
    out = {}
    for g in sorted({key(r) for r in rows}, key=str):
        s = [r for r in rows if key(r) == g]
        out[str(g)] = {"n": len(s), "mean_p": float(np.mean([r["p"] for r in s])), "any_k": float(np.mean([r["anyk"] for r in s])), "gap": float(np.mean([r["anyk"] - r["p"] for r in s]))}
    return out


def test_share() -> dict:
    recs = C.load("results/part4/test")
    n = len(recs)
    dec = [r for r in recs.values() if "P2r" in r["arms"] and any(e["kind"] == "sh" for e in r["arms"]["P2r"]["events"])]
    fail = [r for r in dec if not r["arms"]["P0"]["success"]]
    return {"episodes": n, "with_first_alarm_decision": len(dec), "p0_failed_with_decision": len(fail), "share": len(fail) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/part5/nc/states")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    allrecs = load_states(ROOT / a.dir)
    ok = [r for r in allrecs if r["replay_equal"] and r["d_equal"]]
    dropped = [(r["state"]["pop"], r["state"]["task"], r["state"]["init"], r["state"]["seed_id"]) for r in allrecs if not (r["replay_equal"] and r["d_equal"])]
    A = [r for r in ok if r["state"]["pop"] == "A"]; Cc = [r for r in ok if r["state"]["pop"] == "C"]
    res = {"n_state_files": len(allrecs), "dropped": dropped, "git": sorted({r["git"] for r in allrecs})}
    res["A_R"] = pop_stats(A, "R", a.boot)
    res["A_C"] = pop_stats(A, "C", a.boot)
    res["C_R"] = pop_stats(Cc, "R", a.boot)
    ctrl = [(r["state"]["task"], r["state"]["init"], r["state"]["seed_id"], n, c["equal"]) for r in allrecs if r.get("controls") for n, c in r["controls"].items()]
    res["controls"] = {"checked": len(ctrl), "mismatches": [c for c in ctrl if not c[4]], "states": len({c[:3] for c in ctrl})}
    prim = res["A_R"]
    share = test_share()
    res["test_share"] = share
    g = prim["gap"]["est"]
    res["verdict"] = "GO" if g >= 0.20 else ("WEAK" if g >= 0.10 else "NO-GO")
    res["ceiling_pp_episode"] = {"est": 100 * g * share["share"], "ci": [100 * prim["gap"]["ci"][0] * share["share"], 100 * prim["gap"]["ci"][1] * share["share"]]}
    rows = prim["_rows"]
    res["by_task"] = breakdown(rows, lambda r: r["task"])
    res["by_scene"] = breakdown(rows, lambda r: r["scene"])
    # natural-mix re-weighting (descriptive): weight each task by the number of P0-failed first-alarm GEN cases
    cases, _ = C.gen_cases(C.load("results/part4/gen"))
    nat = {t: sum(1 for c in cases if c["task"] == t and not c["p0"]) for t in range(10)}
    bt = res["by_task"]
    w = {t: nat[int(t)] for t in bt}
    res["gap_natural_mix"] = sum(w[t] * bt[t]["gap"] for t in bt) / sum(w.values())
    res["natural_counts"] = nat
    # preview comparison
    pv = json.load(open(ROOT / "results/part5/nc/preview.json"))
    res["preview_p0_failed"] = pv.get("P0 failed")
    # Spearman of p_s with the frozen Part 4 model's p_retry - p_cont at the state (features from the GEN log)
    try:
        from scipy.stats import spearmanr
        sys.path.insert(0, str(ROOT))
        model = pickle.loads((ROOT / "configs/part4_model.pkl").read_bytes())["sh"]
        cmap = {c["key"]: c for c in cases}
        xs, ps = [], []
        for r in A:
            s_ = r["state"]
            c = cmap.get((s_["task"], s_["init"], s_["seed_id"]))
            if c is None:
                continue
            pr_, pc_ = model.predict(c["x"])
            xs.append(pr_ - pc_); ps.append(float(np.mean([v["success"] for k, v in r["branches"].items() if k.startswith("R")])))
        rho = spearmanr(xs, ps)
        res["spearman_p_vs_model"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(xs)}
    except Exception as e:  # reported, not hidden
        res["spearman_p_vs_model"] = {"error": repr(e)}
    json.dump(res, open(ROOT / "results/part5/nc/results.json", "w"), indent=1, default=float)
    write_md(res)
    print(open(ROOT / "results/part5/notes/p5_0_results.md").read())


def pp(x):
    return f"{100 * x:.1f}"


def write_md(r: dict) -> None:
    P = r["A_R"]
    L = ["# P5.0 noise-ceiling results (Part 4 GEN states; plan results/part5/notes/p5_0_plan.txt)", "",
         f"States: A (P0-failed) {P['n_states']} in {P['n_clusters']} clusters, K = {P['K']} retry draws per state (primary any-of-{P['k_primary']}); dropped (replay mismatch): {r['dropped']}; GEN controls: {r['controls']['checked'] - len(r['controls']['mismatches'])} of {r['controls']['checked']} branches reproduced on {r['controls']['states']} states.", "",
         "## Primary (population A, mode R)", "",
         f"- one random retry (mean p_s, = P2r): **{pp(P['mean_p']['est'])}%** [{pp(P['mean_p']['ci'][0])}, {pp(P['mean_p']['ci'][1])}]",
         f"- perfect noise picker (any-of-{P['k_primary']}): **{pp(P['any_k']['est'])}%** [{pp(P['any_k']['ci'][0])}, {pp(P['any_k']['ci'][1])}]",
         f"- **GAP = {pp(P['gap']['est'])} pp** [{pp(P['gap']['ci'][0])}, {pp(P['gap']['ci'][1])}]  -> **{r['verdict']}** (GO >= 20, WEAK 10-20, NO-GO < 10; decided on the point estimate)",
         f"- episode-level ceiling: GAP x {pp(r['test_share']['share'])}% (Part 4 TEST: {r['test_share']['p0_failed_with_decision']} of {r['test_share']['episodes']} episodes were P0-failed with a first-alarm decision) = **{r['ceiling_pp_episode']['est']:.2f} pp** [{r['ceiling_pp_episode']['ci'][0]:.2f}, {r['ceiling_pp_episode']['ci'][1]:.2f}] over P2r (a bound for a perfect selector, not a forecast for DSRL)", "",
         "## any-of-k curves (success %, [95% CI])", "", "| population / mode | states | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|---|" + "---|" * len(KS)]
    for nm, key in (("A, retry (R)", "A_R"), ("A, continue (C)", "A_C"), ("C (P0 succeeded), retry", "C_R")):
        s = r[key]
        if not s["n_states"]:
            continue
        cells = [f"{pp(s['curve'][k]['est'])} [{pp(s['curve'][k]['ci'][0])}, {pp(s['curve'][k]['ci'][1])}]" if k in s["curve"] else "" for k in KS]
        L.append(f"| {nm} | {s['n_states']} (K={s['K']}) | " + " | ".join(cells) + " |")
    L += ["", "## p_s distribution (share of states)", "", "| population / mode | p_s = 0 | 0 < p_s < 0.5 | p_s >= 0.5 | alarm fires again (all branches) | alarm fires again (rescued branches) |", "|---|---|---|---|---|---|"]
    for nm, key in (("A, R", "A_R"), ("A, C", "A_C"), ("C, R", "C_R")):
        s = r[key]
        if s["n_states"]:
            L.append(f"| {nm} | {pp(s['dist']['zero'])}% | {pp(s['dist']['mid'])}% | {pp(s['dist']['high'])}% | {pp(s['alarm_again_all'])}% | {'-' if s['alarm_again_rescued'] is None else pp(s['alarm_again_rescued']) + '%'} |")
    cs = r["C_R"]
    if cs["n_states"]:
        L += ["", f"Population C (P0 succeeded, {cs['n_states']} states, K = {cs['K']}): a random retry succeeds with p = {pp(cs['mean_p']['est'])}% [{pp(cs['mean_p']['ci'][0])}, {pp(cs['mean_p']['ci'][1])}], i.e. it LOSES the success with probability {pp(1 - cs['mean_p']['est'])}%; the best of {cs['k_primary']} draws succeeds in {pp(cs['any_k']['est'])}%."]
    L += ["", "## Per task (population A, mode R; share of states, mean p_s, any-of-k, gap in pp)", "", "| task | states | mean p_s % | any-of-k % | gap pp |", "|---|---|---|---|---|"]
    for t, v in r["by_task"].items():
        L.append(f"| {t} | {v['n']} | {pp(v['mean_p'])} | {pp(v['any_k'])} | {pp(v['gap'])} |")
    L += ["", "Part 4 TEST: retries paid on tasks 4, 6, 8, 9 (P2r - P0 = +7, +4, +15, +5 pp).", "", "## Scene class at d (label_v2, L6)", "", "| scene | states | mean p_s % | any-of-k % | gap pp |", "|---|---|---|---|---|"]
    for t, v in r["by_scene"].items():
        L.append(f"| {t} | {v['n']} | {pp(v['mean_p'])} | {pp(v['any_k'])} | {pp(v['gap'])} |")
    L += ["", f"GAP re-weighted to the natural task mix of P0-failed GEN cases (descriptive): {pp(r['gap_natural_mix'])} pp.", "",
          f"Spearman of p_s with the frozen Part 4 model's p_retry - p_cont: {r['spearman_p_vs_model']}", "",
          f"GEN any-of-4 preview (P0-failed cases, results/part5/notes/p5_0_preview.md): retry mean p {pp(r['preview_p0_failed']['retry_share'])}%, any-of-4 {pp(r['preview_p0_failed']['any4_retry'])}% (n = {r['preview_p0_failed']['n']}).", "",
          "## Caveats", "", "any-of-K is an UPPER bound with a max-over-draws selection bias; it bounds a perfect selector, not DSRL. The states are a stratified sample (cap 8 per task), not the natural mix. All states are Part 4 GEN states (init 0-19).", ""]
    (ROOT / "results/part5/notes/p5_0_results.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
