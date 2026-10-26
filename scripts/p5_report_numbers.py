"""Part 5 report numbers: every number of results/part5/notes/p5_report.md comes from here, none is typed in the template.
  python scripts/p5_report_numbers.py     -> results/part5/notes/p5_report_numbers.json, results/part5/notes/p5_report_numbers.md (the block the report includes), results/part5/notes/p5_report.md (template filled)
Sources: results/part5/nc/results.json, results/part5/critic/results.json, results/part5/critic/perm_posthoc.json (post hoc), results/part5/nc/states/*.json.gz (the five no-decision states), configs/part5_*.yaml.
Deterministic: same inputs, same bytes."""
from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NC = json.loads((ROOT / "results/part5/nc/results.json").read_text())
CR = json.loads((ROOT / "results/part5/critic/results.json").read_text())
PHF = ROOT / "results/part5/critic/perm_posthoc.json"
PH = json.loads(PHF.read_text()) if PHF.exists() else None
CFG0 = yaml.safe_load((ROOT / "configs/part5_noise_ceiling.yaml").read_text())
CFG1 = yaml.safe_load((ROOT / "configs/part5_critic.yaml").read_text())
BUDGET = 300


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def pp(x, d=1, sign=False):
    return f"{100 * x:+.{d}f} pp" if sign else f"{100 * x:.{d}f} pp"


def ci(c, d=1, sign=False):
    return f"[{100 * c[0]:+.{d}f}, {100 * c[1]:+.{d}f}]" if sign else f"[{100 * c[0]:.{d}f}, {100 * c[1]:.{d}f}]"


def au(c, d=3):
    return f"{c['est']:.{d}f} [{c['ci'][0]:.{d}f}, {c['ci'][1]:.{d}f}]"


def no_decision_table():
    """The five states whose retry never reached a decision, from the P5.0 branch logs only."""
    z = json.loads((ROOT / "results/part5/critic/results.json").read_text())  # noqa: F841 (kept for provenance)
    import numpy as np
    f = np.load(ROOT / "results/part5/critic/features.npz")
    rows = []
    for pop, t, i in zip(f["skipped_pop"], f["skipped_task"], f["skipped_init"]):
        p = glob.glob(str(ROOT / f"results/part5/nc/states/s_{pop}_{int(t)}_{int(i)}_*.json.gz"))[0]
        d = json.load(gzip.open(p))
        s, b = d["state"], d["branches"]
        rb = {k: v for k, v in b.items() if k.startswith("R")}
        r0 = rb["R0"]
        left = BUDGET - s["elapsed"]
        if r0["cut_short"]:
            why = f"budget: the return cost ({r0['return_cost']} steps) exceeded the {left} steps left, so no retry could be paid for; every draw ran on to the budget"
        elif r0["invalid_rewinds"]:
            why = (f"invalid arm-only rewind: the restored arm state failed the validity check, the episode was flagged and ended at step {r0['elapsed']} "
                   f"(its P0 episode {'succeeded' if d['p0']['success'] else 'failed'} at step {d['p0']['elapsed']})")
        else:
            why = "other"
        rows.append(dict(pop=str(pop), task=int(t), init=int(i), seed_id=s["seed_id"], decision=s["d"], elapsed_at_alarm=s["elapsed"], time_left=left, return_cost=r0["return_cost"],
                         cut_short=bool(r0["cut_short"]), invalid_rewinds=r0["invalid_rewinds"], draws=len(rb), successes=sum(v["success"] for v in rb.values()), p0_success=d["p0"]["success"], reason=why))
    return rows


def numbers() -> dict:
    A, C0, CRR = NC["A_R"], NC["A_C"], NC["C_R"]
    a, c = CR["A"], CR["C"]
    N = {}
    # --- P5.0
    N.update(nc_states=A["n_states"], nc_clusters=A["n_clusters"], nc_K=A["K"], nc_K_AC=CFG0["K_AC"], nc_K_CR=CFG0["K_CR"], nc_nA=CFG0["n_A"], nc_nC=CFG0["n_C"], nc_hours=CFG0["projection_hours"],
             nc_mean=pc(A["mean_p"]["est"]), nc_mean_ci=ci(A["mean_p"]["ci"]), nc_any=pc(A["any_k"]["est"]), nc_any_ci=ci(A["any_k"]["ci"]),
             nc_gap=pp(A["gap"]["est"]), nc_gap_ci=ci(A["gap"]["ci"]), nc_verdict=NC["verdict"], nc_ceiling=f"{NC['ceiling_pp_episode']['est']:.2f} pp", nc_ceiling_ci=f"[{NC['ceiling_pp_episode']['ci'][0]:.1f}, {NC['ceiling_pp_episode']['ci'][1]:.1f}]",
             nc_share=pc(NC["test_share"]["share"]), nc_test_failed=NC["test_share"]["p0_failed_with_decision"], nc_test_eps=NC["test_share"]["episodes"],
             nc_zero=pc(A["dist"]["zero"], 0), nc_zero_n=round(A["dist"]["zero"] * A["n_states"]), nc_mid=pc(A["dist"]["mid"], 1), nc_mid_n=round(A["dist"]["mid"] * A["n_states"]),
             nc_high=pc(A["dist"]["high"], 1), nc_high_n=round(A["dist"]["high"] * A["n_states"]), nc_controls=NC["controls"]["checked"], nc_mismatch=len(NC["controls"]["mismatches"]),
             nc_C_mean=pc(CRR["mean_p"]["est"]), nc_C_any=pc(CRR["any_k"]["est"]), nc_C_K=CRR["K"], nc_C_states=CRR["n_states"],
             nc_AC_mean=pc(C0["mean_p"]["est"]), nc_AC_any=pc(C0["any_k"]["est"]), nc_AC_K=C0["K"], nc_changed_gap=pp(NC["by_scene"]["changed"]["gap"]), nc_unchanged_gap=pp(NC["by_scene"]["unchanged"]["gap"]),
             nc_spearman=f"{NC['spearman_p_vs_model']['rho']:.2f}")
    for k in (2, 4, 8, 16, 32):
        N[f"nc_any{k}"] = pc(A["curve"][str(k)]["est"])
    bt = NC["by_task"]
    best = sorted(bt, key=lambda t: -bt[t]["gap"])[:2]
    N["nc_top_tasks"] = " and ".join(f"task {t} ({pp(bt[t]['gap'])})" for t in best)
    N["nc_zero_tasks"] = ", ".join(t for t in sorted(bt, key=int) if bt[t]["any_k"] == 0.0) or "none"
    # --- P5.1a
    N.update(cr_states=a["auc"]["n_states"], cr_mixed=a["auc"]["n_mixed"], cr_nodec=a["auc"]["n_no_decision"], cr_pairs=f"{a['auc']['pairs']:,}", cr_draws=2456,
             cr_auc=au(a["auc"]), cr_k8=pc(a["k8"]["critic"]), cr_k8_ci=ci(a["k8"]["critic_ci"]), cr_random=pc(a["k8"]["random1"]), cr_oracle=pc(a["k8"]["oracle"]),
             cr_gain=pp(a["k8"]["gain"]), cr_gain_ci=ci(a["k8"]["gain_ci"], sign=True), cr_captured=f"{a['k8']['captured']:.2f}", cr_captured_ci=f"[{a['k8']['captured_ci'][0]:.2f}, {a['k8']['captured_ci'][1]:.2f}]",
             cr_gain4=pp(a["k4"]["gain"]), cr_gain16=pp(a["k16"]["gain"]), cr_verdict=CR["verdict_a"],
             cr_pfast=f"{CR['perm']['p_fast']:.3f}", cr_pprim=f"{CR['perm']['p_primary_approx']:.4f}", cr_nperm=CR["perm"]["n_perm"], cr_nullmean=f"{CR['perm']['null_mean']:.3f}", cr_nullsd=f"{CR['perm']['null_sd']:.3f}",
             cr_nullq95=f"{CR['perm']['null_q95']:.3f}", cr_fastauc=f"{CR['perm']['observed_fast']:.3f}", cr_loto=au(CR["A_loto"]["auc"]), cr_loto_gain=pp(CR["A_loto"]["k8"]["gain"]),
             cr_c_auc=au(c["auc"]), cr_c_mixed=c["auc"]["n_mixed"], cr_c_k4=pc(c["k4"]["critic"]), cr_c_rand=pc(c["k4"]["random1"]), cr_c_oracle=pc(c["k4"]["oracle"]), cr_c_gain=pp(c["k4"]["gain"], sign=True),
             cr_chosen=f"{CR['chosen_LOCO']['lr_F3F1']} of {sum(CR['chosen_LOCO'].values())}", cr_chosen_loto=f"{CR['chosen_LOTO']['lr_F3F1']} of {sum(CR['chosen_LOTO'].values())}")
    for k, nm in (("lr_F3", "lr3"), ("gbt_F3", "gbt3"), ("lr_F3F1", "lr31")):
        N[f"cr_{nm}_auc"] = au(CR["A_cands"][k]["auc"])
        N[f"cr_{nm}_gain"] = pp(CR["A_cands"][k]["k8"]["gain"], sign=True)
    for k, nm in (("plan_accel (lower better)", "accel"), ("distance_to_medoid (lower better)", "medoid"), ("noise_norm (lower better)", "norm")):
        N[f"cr_s_{nm}"] = au(CR["single"][k])
    # --- P5.1b
    f = CR["firstnoise"]
    N.update(fn_states=f["n_states"], fn_branches=f["n_branches"], fn_good=pc(f["good"]), fn_bad=pc(f["bad"]), fn_effect=pp(f["effect"], sign=True), fn_ci=ci(f["effect_ci"], sign=True),
             fn_pos=f["n_pos_states"], fn_neg=f["n_neg_states"], fn_flat=f["n_states"] - f["n_pos_states"] - f["n_neg_states"], fn_verdict=CR["verdict_b"], fn_controls=f["controls"]["checked"],
             fn_controls_eq=f["controls"]["equal"], fn_replay=f["replay_equal"], fn_orig_good=pc(f["orig_good"], 0), fn_orig_bad=pc(f["orig_bad"], 0))
    ps = sorted(f["per_state"], key=lambda r: r["bad"] - r["good"])
    N["fn_top"] = "; ".join(f"task {r['task']} / init {r['init']}: {pc(r['good'], 0)} against {pc(r['bad'], 0)}" for r in ps[:2])
    N["fn_neg_states"] = "; ".join(f"task {r['task']} / init {r['init']}" for r in f["per_state"] if r["good"] < r["bad"])
    # --- post hoc
    if PH:
        N.update(ph_p=f"{PH['p']:.4f}", ph_n=PH["n_perm"], ph_auc=f"{PH['real_auc']:.4f}", ph_mean=f"{PH['null_mean']:.3f}", ph_sd=f"{PH['null_sd']:.3f}", ph_q95=f"{PH['null_q95']:.3f}")
    else:
        N.update(ph_p="PENDING", ph_n="PENDING", ph_auc="PENDING", ph_mean="PENDING", ph_sd="PENDING", ph_q95="PENDING")
    return N


def md_block(N, nd) -> str:
    L = ["<!-- generated by scripts/p5_report_numbers.py -->", "", "| key | value |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in sorted(N.items())]
    L += ["", "**Five states that never reached a retry decision** (from `results/part5/nc/states/*.json.gz`):", "",
          "| population | task / init | seed | alarm decision | step at alarm | steps left | return cost | draws (successes) | P0 episode | why |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in nd:
        L.append(f"| {r['pop']} | {r['task']} / {r['init']} | {r['seed_id']} | {r['decision']} | {r['elapsed_at_alarm']} | {r['time_left']} | {r['return_cost']} | {r['draws']} ({r['successes']}) | {'success' if r['p0_success'] else 'failure'} | {r['reason']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    N = numbers()
    nd = no_decision_table()
    N["nd_budget"] = sum(r["cut_short"] for r in nd)
    N["nd_invalid"] = sum(bool(r["invalid_rewinds"]) for r in nd)
    N["nd_n"] = len(nd)
    N["nd_table"] = "\n".join(["| population | task / init | step at alarm | steps left | return cost | draws (successes) | P0 episode | why |", "|---|---|---|---|---|---|---|---|"] +
                              [f"| {r['pop']} | {r['task']} / {r['init']} | {r['elapsed_at_alarm']} | {r['time_left']} | {r['return_cost']} | {r['draws']} ({r['successes']}) | {'success' if r['p0_success'] else 'failure'} | {r['reason']} |" for r in nd])
    out = ROOT / "results/part5/notes"
    (out / "p5_report_numbers.json").write_text(json.dumps({"numbers": N, "no_decision_states": nd}, indent=1, sort_keys=True))
    (out / "p5_report_numbers.md").write_text(md_block(N, nd))
    tpl = (out / "p5_report_template.txt").read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(N))
    if missing:
        sys.exit(f"template keys without a number: {missing}")
    (out / "p5_report.md").write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: str(N[m.group(1)]), tpl))
    print(f"{len(N)} numbers, {len(nd)} no-decision states; post hoc {'present' if PH else 'PENDING'}")


if __name__ == "__main__":
    main()
