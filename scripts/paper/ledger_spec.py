"""The ledger of numbers the paper may cite. One entry per number: where it lives (file + key path or a named derivation), its CI and n, the part, and its status
(pre-registered | descriptive | post hoc). `scripts/paper/build_ledger.py` resolves every entry from the committed / manifested result files;
`tests/paper/test_ledger.py` re-resolves them independently."""
from __future__ import annotations

D1 = "results/part1/deltadino/seed{s}/eval.json"
V2 = "results/part1/deltadino_v2/seed0/eval.json"
CAL = "results/part2/cal_report.json"
P2 = "results/part2/results.json"
P3 = "results/part3/results_test.json"
P4 = "results/part4/results_test.json"
NC = "results/part5/nc/results.json"
CR = "results/part5/critic/results.json"
PH = "results/part5/critic/perm_posthoc.json"
PRE, DESC, POST = "pre-registered", "descriptive", "post hoc"

ROWS: list[dict] = []


def add(id, name, part, file, path, status, ci=None, n=None, unit="frac", note="", derive=None):
    """n: an int or a (file, path) pair; ci: a key path (tuple) in the same file, or a pair of key paths (lo, hi)."""
    ROWS.append(dict(id=id, name=name, part=part, file=file, path=tuple(path), ci=None if ci is None else tuple(ci), n=n, unit=unit, status=status, note=note, derive=derive))


# ---------------------------------------------------------------- Part 1 (DeltaDINO v1, SPEC section 7.2)
for s in (0, 1, 2):
    add(f"p1.test_auroc.seed{s}", f"DeltaDINO test AUROC, seed {s}", "1", D1.format(s=s), ("test", "image", "auroc"), PRE, ci=("test", "image", "auroc_ci"), n=("test", "image", "n"))
add("p1.test_auroc.mean3", "DeltaDINO test AUROC, mean of 3 seeds (CI = min-max over seeds)", "1", D1, ("test", "image", "auroc"), PRE, n=("test", "image", "n"), derive="mean3_range")
add("p1.heldout_auroc.seed0", "DeltaDINO AUROC on held-out tasks (6, 8), seed 0", "1", D1.format(s=0), ("test_heldout_tasks", "image", "auroc"), DESC, ci=("test_heldout_tasks", "image", "auroc_ci"), n=("test_heldout_tasks", "image", "n"))
add("p1.policy_auroc.seed0", "DeltaDINO AUROC on policy pairs (test_policy), seed 0", "1", D1.format(s=0), ("test_policy", "image", "auroc"), DESC, ci=("test_policy", "image", "auroc_ci"), n=("test_policy", "image", "n"))
add("p1.hardneg_fa.seed0", "DeltaDINO false-alarm rate on hard negatives H1+H2 (test), seed 0", "1", D1.format(s=0), ("test", "hard_neg", "H1+H2", "false_alarm"), PRE, n=("test", "hard_neg", "H1+H2", "n"))
add("p1.bigchange_recall.seed0", "DeltaDINO recall on large changes (>2 cm / >15 deg), seed 0", "1", D1.format(s=0), ("test", "big_change_recall", "recall"), PRE, n=("test", "big_change_recall", "n"))
add("p15.v2_test_auroc", "DeltaDINO-v2 (Part 1.5) test AUROC, seed 0", "1.5", V2, ("test", "image", "auroc"), DESC, ci=("test", "image", "auroc_ci"), n=("test", "image", "n"))
add("p15.v2_policy_auroc", "DeltaDINO-v2 (Part 1.5) AUROC on policy pairs, seed 0", "1.5", V2, ("test_policy", "image", "auroc"), DESC, ci=("test_policy", "image", "auroc_ci"), n=("test_policy", "image", "n"))

# ---------------------------------------------------------------- Part 2 (CAL, in-loop gate, DEV/TEST arms on the seen tasks)
add("p2.cal.catch_rate", "Alarm catch rate: failed CAL episodes alarmed before timeout (accel CUSUM)", "2", CAL, ("detectors", "accel", "failed_alarmed_frac"), DESC, n=("detectors", "accel", "n_failed"), note="")
add("p2.cal.false_alarm", "Alarm false-alarm rate on successful CAL episodes (accel CUSUM, alpha = 0.1)", "2", CAL, ("detectors", "accel", "realized_fpr_in_sample"), PRE, n=("success_episodes",))
add("p2.cal.first_alarm_step", "First alarm, median step (accel); IQR in the note", "2", CAL, ("detectors", "accel", "first_alarm_step", "median"), DESC, n=("detectors", "accel", "failed_alarmed"), unit="steps")
add("p2.cal.lead_budget", "Lead time: median budget left at the first alarm, steps (of 300)", "2", CAL, ("detectors", "accel", "budget_left_at_alarm_steps", "median"), DESC, n=("detectors", "accel", "failed_alarmed"), unit="steps")
add("p2.cal.lead_ge150", "Share of first alarms leaving >= 150 steps", "2", CAL, ("detectors", "accel", "alarms_leaving_at_least_steps", "150"), DESC, n=("detectors", "accel", "failed_alarmed"))
add("p2.gate.auroc_seen", "In-loop gate AUROC, seen tasks", "2", P2, ("gate_in_loop", "seen", "auroc"), DESC, n=("gate_in_loop", "seen", "n"))
add("p2.gate.auroc_all", "In-loop gate AUROC, all tasks (incl. held-out 6, 8)", "2", P2, ("gate_in_loop", "all", "auroc"), DESC, n=("gate_in_loop", "all", "n"))
add("p2.gate.recall_seen", "In-loop gate recall, seen tasks", "2", P2, ("gate_in_loop", "seen", "recall"), DESC, n=("gate_in_loop", "seen", "changed"))
add("p2.gate.fpr_seen", "In-loop gate false-positive rate, seen tasks", "2", P2, ("gate_in_loop", "seen", "fpr"), DESC, n=("gate_in_loop", "seen", "unchanged"))
for arm in ("P0", "P1", "P2", "P3", "P4", "P5", "P6"):
    T = ("tables", "seen", "arms", arm)
    add(f"p2.seen.{arm}.S", f"Part 2 seen, {arm}: success", "2", P2, T + ("S", "est"), DESC, ci=T + ("S", "ci"), n=("tables", "seen", "n_episodes"))
    add(f"p2.seen.{arm}.SF", f"Part 2 seen, {arm}: silent failures", "2", P2, T + ("SF", "est"), DESC, ci=T + ("SF", "ci"), n=("tables", "seen", "n_episodes"))
    add(f"p2.seen.{arm}.F", f"Part 2 seen, {arm}: flagged episodes", "2", P2, T + ("F", "est"), DESC, ci=T + ("F", "ci"), n=("tables", "seen", "n_episodes"))
    if arm != "P0":
        add(f"p2.seen.{arm}.rescue", f"Part 2 seen, {arm}: rescues (P0 failed, arm succeeded), episodes", "2", P2, T + ("rescue", "count"), DESC, unit="count", n=("tables", "seen", "n_episodes"))
        add(f"p2.seen.{arm}.harm", f"Part 2 seen, {arm}: harms (P0 succeeded, arm failed), episodes", "2", P2, T + ("harm", "count"), DESC, unit="count", n=("tables", "seen", "n_episodes"))
add("p2.seen.P3.FP", "Part 2 seen, P3: flag precision (flagged episodes that failed in P0)", "2", P2, ("tables", "seen", "arms", "P3", "FP", "est"), DESC, n=("tables", "seen", "arms", "P3", "FP", "flags"), note="")
add("p2.H1", "Part 2 H1: S(P3) - S(P0) (not supported)", "2", P2, ("hypotheses", "seen", "H1", "diff"), PRE, ci=("hypotheses", "seen", "H1", "ci95"), n=("tables", "seen", "n_episodes"))
add("p2.H2.SF", "Part 2 H2: SF(P3) - SF(P2) (not supported)", "2", P2, ("hypotheses", "seen", "H2", "SF_diff"), PRE, ci=("hypotheses", "seen", "H2", "SF_ci95"), n=("tables", "seen", "n_episodes"))
add("p2.H2.S", "Part 2 H2: S(P3) - S(P2), the success side of H2", "2", P2, ("hypotheses", "seen", "H2", "S_diff"), PRE, ci=("hypotheses", "seen", "H2", "S_ci95"), n=("tables", "seen", "n_episodes"))
add("p2.H3", "Part 2 H3: S(P3) - S(P5), random-timed (supported)", "2", P2, ("hypotheses", "seen", "H3", "diff"), PRE, ci=("hypotheses", "seen", "H3", "ci95"), n=("tables", "seen", "n_episodes"))

# ---------------------------------------------------------------- Part 3 (TEST, PA = grace on the first CHANGED verdict)
for arm in ("P0", "P3", "PA", "PAc"):
    for pop, n in (("seen", ("tables", "seen", "n_episodes")), ("all", ("tables", "all", "n_episodes"))):
        T = ("tables", pop, "arms", arm)
        st = PRE if pop == "seen" else DESC
        add(f"p3.{pop}.{arm}.S", f"Part 3 TEST {pop}, {arm}: success", "3", P3, T + ("S", "est"), st, ci=T + ("S", "ci"), n=n)
        add(f"p3.{pop}.{arm}.SF", f"Part 3 TEST {pop}, {arm}: silent failures", "3", P3, T + ("SF", "est"), st, ci=T + ("SF", "ci"), n=n)
        add(f"p3.{pop}.{arm}.F", f"Part 3 TEST {pop}, {arm}: flagged episodes", "3", P3, T + ("F", "est"), DESC, ci=T + ("F", "ci"), n=n)
        if arm != "P0":
            add(f"p3.{pop}.{arm}.rescue", f"Part 3 TEST {pop}, {arm}: rescues, episodes", "3", P3, T + ("rescue", "count"), DESC, unit="count", n=n)
            add(f"p3.{pop}.{arm}.harm", f"Part 3 TEST {pop}, {arm}: harms, episodes", "3", P3, T + ("harm", "count"), DESC, unit="count", n=n)
for h, nm in (("H1", "S(PA) - S(P3) (supported)"), ("H2", "SF(P0) - SF(PA), margin 10 pp (not supported)"), ("H3", "S(PA) - S(P0), non-inferiority margin -3 pp (not supported)")):
    add(f"p3.{h}", f"Part 3 {h} (seen): {nm}", "3", P3, ("hypotheses", "seen", h, "est"), PRE, ci=("hypotheses", "seen", h, "ci95"), n=("tables", "seen", "n_episodes"))
    add(f"p3.{h}.p_holm", f"Part 3 {h} (seen): Holm-adjusted p", "3", P3, ("hypotheses", "seen", h, "p_holm"), PRE, n=("tables", "seen", "n_episodes"), unit="p")

# ---------------------------------------------------------------- Part 4 (TEST, 1000 episodes)
ARMS4 = ["P0", "PC", "P2r", "SH", "SH-task", "SH-noG", "SH+F"]
for arm in ARMS4:
    T = ("tables", "all", "arms", arm)
    n = ("tables", "all", "n_episodes")
    add(f"p4.{arm}.S", f"Part 4 TEST, {arm}: success", "4", P4, T + ("S", "est"), DESC, ci=T + ("S", "ci"), n=n)
    add(f"p4.{arm}.SF", f"Part 4 TEST, {arm}: silent failures", "4", P4, T + ("SF", "est"), DESC, ci=T + ("SF", "ci"), n=n)
    add(f"p4.{arm}.F", f"Part 4 TEST, {arm}: flagged episodes", "4", P4, T + ("F", "est"), DESC, ci=T + ("F", "ci"), n=n)
    add(f"p4.{arm}.retry_rate", f"Part 4 TEST, {arm}: episodes retried", "4", P4, T + ("retry_rate",), DESC, n=n)
    if arm != "P0":
        add(f"p4.{arm}.rescue", f"Part 4 TEST, {arm}: rescues, episodes", "4", P4, T + ("rescue", "count"), DESC, unit="count", n=n)
        add(f"p4.{arm}.harm", f"Part 4 TEST, {arm}: harms, episodes", "4", P4, T + ("harm", "count"), DESC, unit="count", n=n)
add("p4.SH+F.FP", "Part 4 TEST, SH+F: flag precision", "4", P4, ("tables", "all", "arms", "SH+F", "FP", "est"), DESC, ci=("tables", "all", "arms", "SH+F", "FP", "ci"), n=("tables", "all", "arms", "SH+F", "FP", "flags"), note="flags that were P0 failures")
for h, nm in (("H1", "S(SH) - S(P0) (supported)"), ("H2", "S(SH) - S(P2r) (not supported)"), ("H3", "S(SH) - S(SH-task) (not supported)")):
    add(f"p4.{h}", f"Part 4 {h}: {nm}", "4", P4, ("hypotheses", "all", h, "est"), PRE, ci=("hypotheses", "all", h, "ci95"), n=("tables", "all", "n_episodes"))
    add(f"p4.{h}.p_holm", f"Part 4 {h}: Holm-adjusted p", "4", P4, ("hypotheses", "all", h, "p_holm"), PRE, n=("tables", "all", "n_episodes"), unit="p")

# ---------------------------------------------------------------- Part 5.0 (noise ceiling) and 5.1 (critic)
add("p5.0.mean_p", "P5.0: one random retry rescues (mean over 40 P0-failed states)", "5.0", NC, ("A_R", "mean_p", "est"), PRE, ci=("A_R", "mean_p", "ci"), n=("A_R", "n_states"))
add("p5.0.any64", "P5.0: any of 64 retry draws rescues", "5.0", NC, ("A_R", "any_k", "est"), PRE, ci=("A_R", "any_k", "ci"), n=("A_R", "n_states"))
add("p5.0.gap", "P5.0: GAP = any-of-64 minus one random retry (GO if >= pre-registered threshold)", "5.0", NC, ("A_R", "gap", "est"), PRE, ci=("A_R", "gap", "ci"), n=("A_R", "n_states"))
for k in (2, 4, 8, 16, 32):
    add(f"p5.0.any{k}", f"P5.0: any of {k} retry draws rescues", "5.0", NC, ("A_R", "curve", k, "est"), DESC, ci=("A_R", "curve", k, "ci"), n=("A_R", "n_states"))
add("p5.0.ceiling_episode", "P5.0: episode-level ceiling over P2r, pp of TEST episodes", "5.0", NC, ("ceiling_pp_episode", "est"), PRE, ci=("ceiling_pp_episode", "ci"), n=("test_share", "episodes"), unit="pp")
add("p5.0.dist_zero", "P5.0: states that never succeed in 64 draws ('habit')", "5.0", NC, ("A_R", "dist", "zero"), DESC, n=("A_R", "n_states"))
add("p5.0.dist_mid", "P5.0: states with 0 < p_s < 0.5 ('sometimes')", "5.0", NC, ("A_R", "dist", "mid"), DESC, n=("A_R", "n_states"))
add("p5.0.dist_high", "P5.0: states with p_s >= 0.5 ('usually')", "5.0", NC, ("A_R", "dist", "high"), DESC, n=("A_R", "n_states"))
add("p5.0.C.mean_p", "P5.0 harm side (C, P0-succeeded states): one random retry succeeds", "5.0", NC, ("C_R", "mean_p", "est"), DESC, ci=("C_R", "mean_p", "ci"), n=("C_R", "n_states"))
add("p5.0.C.any8", "P5.0 harm side: any of 8 draws succeeds", "5.0", NC, ("C_R", "any_k", "est"), DESC, ci=("C_R", "any_k", "ci"), n=("C_R", "n_states"))
add("p5.1a.auc", "P5.1a: primary critic within-state AUROC (held-out A states, leave-one-cluster-out)", "5.1", CR, ("A", "auc", "est"), PRE, ci=("A", "auc", "ci"), n=("A", "auc", "pairs"), note="n = (success, failure) pairs of 23 mixed states")
add("p5.1a.k8_critic", "P5.1a: critic best-of-8 success", "5.1", CR, ("A", "k8", "critic"), PRE, ci=("A", "k8", "critic_ci"), n=("A", "auc", "n_states"))
add("p5.1a.k8_oracle", "P5.1a: oracle any-of-8 success", "5.1", CR, ("A", "k8", "oracle"), DESC, ci=("A", "k8", "oracle_ci"), n=("A", "auc", "n_states"))
add("p5.1a.k8_gain", "P5.1a: critic best-of-8 minus random-of-1", "5.1", CR, ("A", "k8", "gain"), PRE, ci=("A", "k8", "gain_ci"), n=("A", "auc", "n_states"))
add("p5.1a.captured", "P5.1a: share of the oracle gap captured at k = 8", "5.1", CR, ("A", "k8", "captured"), DESC, ci=("A", "k8", "captured_ci"), n=("A", "auc", "n_states"))
add("p5.1a.p_perm", "P5.1a: permutation p, fast pipeline (pre-registered null)", "5.1", CR, ("perm", "p_fast"), PRE, n=("perm", "n_perm"), unit="p")
add("p5.1a.p_perm_primary_approx", "P5.1a: primary AUROC against the fast-pipeline null (approximate)", "5.1", CR, ("perm", "p_primary_approx"), PRE, n=("perm", "n_perm"), unit="p")
add("p5.1a.loto_auc", "P5.1a: primary critic, leave-one-task-out AUROC", "5.1", CR, ("A_loto", "auc", "est"), DESC, ci=("A_loto", "auc", "ci"), n=("A_loto", "auc", "pairs"))
for k, nm in (("plan_accel (lower better)", "plan acceleration"), ("distance_to_medoid (lower better)", "distance to the medoid plan"), ("noise_norm (lower better)", "noise norm")):
    add(f"p5.1a.single.{k.split(' ')[0]}", f"P5.1a: single feature AUROC, {nm}", "5.1", CR, ("single", k, "est"), DESC, ci=("single", k, "ci"), n=("single", k, "pairs"))
add("p5.1a.C.auc", "P5.1a harm side (C): critic within-state AUROC", "5.1", CR, ("C", "auc", "est"), DESC, ci=("C", "auc", "ci"), n=("C", "auc", "pairs"))
add("p5.1a.C.k4_gain", "P5.1a harm side (C): critic best-of-4 minus random-of-1", "5.1", CR, ("C", "k4", "gain"), DESC, ci=("C", "k4", "gain_ci"), n=("C", "auc", "n_states"))
add("p5.1b.effect", "P5.1b: first-noise effect, P(success | first noise from a successful draw) - P(... failed draw)", "5.1", CR, ("firstnoise", "effect"), PRE, ci=("firstnoise", "effect_ci"), n=("firstnoise", "n_branches"))
add("p5.1b.good", "P5.1b: success with a first noise taken from a successful P5.0 draw", "5.1", CR, ("firstnoise", "good"), PRE, n=("firstnoise", "n_branches"))
add("p5.1b.bad", "P5.1b: success with a first noise taken from a failed P5.0 draw", "5.1", CR, ("firstnoise", "bad"), PRE, n=("firstnoise", "n_branches"))
add("p5.1b.n_pos", "P5.1b: states with a positive effect (of 10)", "5.1", CR, ("firstnoise", "n_pos_states"), DESC, unit="count", n=("firstnoise", "n_states"))
add("p5.1b.n_neg", "P5.1b: states with a negative effect (of 10)", "5.1", CR, ("firstnoise", "n_neg_states"), DESC, unit="count", n=("firstnoise", "n_states"))
add("p5.1.posthoc.p", "POST HOC: permutation p with the actual primary critic (lr_F3F1), 1000 within-state label shuffles", "5.2", PH, ("p",), POST, n=("n_perm",), unit="p")
add("p5.1.posthoc.null_mean", "POST HOC: null mean AUROC, actual primary critic", "5.2", PH, ("null_mean",), POST, n=("n_perm",))
add("p5.1.posthoc.null_q95", "POST HOC: null 95th percentile AUROC, actual primary critic", "5.2", PH, ("null_q95",), POST, n=("n_perm",))

# numbers quoted in chat or in earlier summaries, checked against the ledger: (quote, ledger id, quoted value, decimals, unit scale, population note)
QUOTES = [
    ("86% of failed episodes alarmed", "p2.cal.catch_rate", 86, 0, 100, "CAL, accel CUSUM"),
    ("AUROC 0.99 (gate)", "p2.gate.auroc_seen", 0.99, 2, 1, "in-loop gate on SEEN tasks (see p2.gate.auroc_all and p1.test_auroc.mean3 for the other populations)"),
    ("silent failures 32% (P0)", "p2.seen.P0.SF", 32, 0, 100, "Part 2 seen tasks; TEST populations differ (p3.all.P0.SF, p4.P0.SF)"),
    ("silent failures 15% (P3)", "p2.seen.P3.SF", 15, 0, 100, "Part 2 seen tasks (p3.all.P3.SF for Part 3 TEST)"),
    ("success cost -2.9 pp (P3 vs P0)", "p2.H1", -2.9, 1, 100, "Part 2 seen"),
    ("H1 +2.1 pp (SH vs P0)", "p4.H1", 2.1, 1, 100, "Part 4 TEST"),
    ("SH+F precision 86.7%", "p4.SH+F.FP", 86.7, 1, 100, "flags that were P0 failures"),
    ("P5.0 GAP 31.7 pp", "p5.0.gap", 31.7, 1, 100, "40 states"),
    ("P5.1 AUROC 0.552", "p5.1a.auc", 0.552, 3, 1, "primary critic"),
    ("P5.1 best-of-8 gain +2.4 pp", "p5.1a.k8_gain", 2.4, 1, 100, "A states"),
    ("first-noise effect +18.8 pp", "p5.1b.effect", 18.8, 1, 100, "10 states"),
]

# signed differences of fractions are shown in percentage points
DIFF_IDS = {"p2.H1", "p2.H2.SF", "p2.H2.S", "p2.H3", "p3.H1", "p3.H2", "p3.H3", "p4.H1", "p4.H2", "p4.H3", "p5.0.gap", "p5.1a.k8_gain", "p5.1a.C.k4_gain", "p5.1b.effect"}
for r in ROWS:
    if r["id"] in DIFF_IDS:
        r["unit"] = "diff"
AUC_IDS = {r["id"] for r in ROWS if any(t in r["id"] for t in ("auroc", ".auc", "loto_auc", ".single.", "null_mean", "null_q95"))}
for r in ROWS:
    if r["id"] in AUC_IDS:
        r["unit"] = "auc"
