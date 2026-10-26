"""paper/tables/T1..T4 as CSV + LaTeX, from paper/numbers.json (the ledger) and the Part 5 result files. Rerunnable; deterministic.
  python scripts/paper/build_tables.py"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PAPER, load  # noqa: E402

LED = {r["id"]: r for r in json.loads((PAPER / "numbers.json").read_text())["rows"]}
OUT = PAPER / "tables"


def pct(i):
    r = LED[i]
    return f"{100 * r['value']:.1f}" if r["ci"] is None else f"{100 * r['value']:.1f} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"


def cnt(i):
    return str(int(LED[i]["value"]))


def tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write(name, header, rows, caption, align=None):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    align = align or "l" * 2 + "r" * (len(header) - 2)
    L = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex_escape(caption)}}}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
         " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    L += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}", rf"\label{{tab:{name}}}", r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L) + "\n")


def t1():
    rows = []
    for arm, desc in (("P0", "do nothing"), ("P1", "re-roll noise at the alarm (no pause, no rewind)"), ("P2", "blind retry (rewind, no check)"), ("P3", "Second Look: pause, camera check, flag if changed, else retry"),
                      ("P5", "as P3 with random-timed alarms"), ("P6", "check only (flag if changed, else continue)")):
        a = f"p2.seen.{arm}"
        rows.append(["2", arm, desc, "seen tasks, 720", pct(a + ".S"), pct(a + ".SF"), pct(a + ".F")])
    for arm, desc in (("P0", "do nothing"), ("P3", "Second Look (as Part 2)"), ("PA", "P3 with a grace on the first CHANGED verdict"), ("PAc", "PA, later alarms take the P3 path")):
        a = f"p3.all.{arm}"
        rows.append(["3", arm, desc, "TEST, 1000", pct(a + ".S"), pct(a + ".SF"), pct(a + ".F")])
    for arm, desc in (("P0", "do nothing"), ("PC", "pause, then continue"), ("P2r", "always retry once"), ("SH", "retry iff the value model says so"), ("SH-task", "SH from the task id only"),
                      ("SH-noG", "SH without the camera feature"), ("SH+F", "SH, and flag low-value cases")):
        a = f"p4.{arm}"
        rows.append(["4", arm, desc, "TEST, 1000", pct(a + ".S"), pct(a + ".SF"), pct(a + ".F")])
    write("T1_policy_comparison", ["Part", "Arm", "What it does", "Population, episodes", "Success % [95% CI]", "Silent failures % [95% CI]", "Flagged %"], rows,
          "Success and silent failures per policy across Parts 2-4. Populations differ by part (Part 2 seen tasks; Parts 3-4 TEST, init 20-39) so only arms within a part are paired; "
          "P0 is each part's own baseline. Flagged = episodes ended by a flag.", "lllrrrr")


def t2():
    rows = []
    spec = [("2", "P1", "p2.seen.P1", None), ("2", "P2", "p2.seen.P2", None), ("2", "P3", "p2.seen.P3", None), ("2", "P5", "p2.seen.P5", None), ("2", "P6", "p2.seen.P6", None),
            ("3", "P3", "p3.all.P3", None), ("3", "PA", "p3.all.PA", None), ("3", "PAc", "p3.all.PAc", None)] + [("4", a, f"p4.{a}", f"p4.{a}.retry_rate") for a in ARMS4]
    n = {"2": 720, "3": 1000, "4": 1000}
    for part, arm, base, rr in spec:
        res, har = int(LED[base + ".rescue"]["value"]), int(LED[base + ".harm"]["value"])
        rows.append([part, arm, str(n[part]), str(res), str(har), f"{res - har:+d}", f"{100 * LED[rr]['value']:.1f}" if rr else "-"])
    write("T2_rescue_harm", ["Part", "Arm", "Episodes", "Rescues", "Harms", "Net", "Retried %"], rows,
          "Rescue (P0 failed, the arm succeeded) and harm (P0 succeeded, the arm failed) counts against the same episodes' P0 outcome, per intervention. "
          "Blind retry rescues and harms about equally; the learned model (SH) retries less and keeps the same net.", "llrrrrr")


ARMS4 = ["PC", "P2r", "SH", "SH-task", "SH-noG", "SH+F"]


def t3():
    nc = load("results/part5/nc/results.json")
    A = nc["A_R"]
    rows = [[str(k), f"{100 * A['curve'][str(k)]['est']:.1f}", f"{100 * A['curve'][str(k)]['ci'][0]:.1f}", f"{100 * A['curve'][str(k)]['ci'][1]:.1f}"] for k in (1, 2, 4, 8, 16, 32, 64)]
    write("T3a_any_of_k", ["k draws", "Rescued % (any of k)", "CI low", "CI high"], rows,
          f"P5.0: share of the {A['n_states']} P0-failed first-alarm states rescued by at least one of k retry draws (k = 1 is one random retry). Cluster bootstrap over (task, init).", "rrrr")
    bt = nc["by_task"]
    rows = [[t, str(bt[t]["n"]), f"{100 * bt[t]['mean_p']:.1f}", f"{100 * bt[t]['any_k']:.1f}", f"{100 * bt[t]['gap']:.1f}"] for t in sorted(bt, key=int)]
    write("T3b_per_task", ["Task", "States", "One random retry %", "Any of 64 %", "Gap pp"], rows,
          "P5.0 per task (descriptive; 3-5 states per task). Tasks 2 and 9 are habit-like (no or few successes), tasks 4 and 8 retry well even without steering.", "rrrrr")


def t4():
    cr = load("results/part5/critic/results.json")
    rows = []
    for key, nm in (("lr_F3", "logistic, plan features F3"), ("gbt_F3", "boosted trees, F3"), ("lr_F3F1", "logistic, F3 + noise (primary)")):
        c = cr["A_cands"][key]
        rows.append([nm, f"{c['auc']['est']:.3f} [{c['auc']['ci'][0]:.3f}, {c['auc']['ci'][1]:.3f}]", f"{100 * c['k8']['gain']:+.1f} [{100 * c['k8']['gain_ci'][0]:+.1f}, {100 * c['k8']['gain_ci'][1]:+.1f}]"])
    c = cr["A_loto"]
    rows.append(["primary, leave-one-task-out (descriptive)", f"{c['auc']['est']:.3f} [{c['auc']['ci'][0]:.3f}, {c['auc']['ci'][1]:.3f}]", f"{100 * c['k8']['gain']:+.1f} [{100 * c['k8']['gain_ci'][0]:+.1f}, {100 * c['k8']['gain_ci'][1]:+.1f}]"])
    for k, nm in (("plan_accel (lower better)", "single feature: plan acceleration"), ("distance_to_medoid (lower better)", "single feature: distance to medoid plan"), ("noise_norm (lower better)", "single feature: noise norm")):
        s = cr["single"][k]
        rows.append([nm, f"{s['est']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]", "-"])
    c = cr["C"]
    rows.append(["harm side (C states), primary critic", f"{c['auc']['est']:.3f} [{c['auc']['ci'][0]:.3f}, {c['auc']['ci'][1]:.3f}]", f"{100 * c['k4']['gain']:+.1f} [{100 * c['k4']['gain_ci'][0]:+.1f}, {100 * c['k4']['gain_ci'][1]:+.1f}] (best-of-4)"])
    write("T4_critic", ["Critic (held-out A states)", "Within-state AUROC [95% CI]", "Best-of-8 minus random-of-1, pp [95% CI]"], rows,
          f"P5.1a: can a critic rank a state's successful retry draws above its failed ones? Within-state AUROC pooled over {cr['A']['auc']['n_mixed']} mixed states ({cr['A']['auc']['pairs']:,} pairs); leave-one-cluster-out. "
          "Pre-registered rule: GO needs AUROC >= 0.65 and a gain >= 10 pp; the primary critic is the logistic model with noise features.", "lrr")


def main():
    t1(); t2(); t3(); t4()
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
