"""paper/claims.md: the claims the paper will make, each with the ledger rows that back it and its status. Numbers are filled from paper/numbers.json.
  python scripts/paper/build_claims.py"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PAPER  # noqa: E402

LED = {r["id"]: r for r in json.loads((PAPER / "numbers.json").read_text())["rows"]}

# (title, claim text with {row id} placeholders that expand to the displayed value, status, caveat)
CLAIMS = [
    ("C1. DeltaDINO sees scene changes",
     "A fine-tuned DINO-based scene-change detector separates changed from unchanged scenes offline (test AUROC {p1.test_auroc.mean3}, three seeds; hard-negative false alarms {p1.hardneg_fa.seed0}) "
     "and inside the closed loop on seen tasks (AUROC {p2.gate.auroc_seen}, recall {p2.gate.recall_seen}, false-positive rate {p2.gate.fpr_seen}); it is weaker on held-out tasks (offline {p1.heldout_auroc.seed0}) and when they are pooled in the loop ({p2.gate.auroc_all}).",
     "confirmatory for the offline test criterion (pre-registered, Part 1); descriptive for the in-loop and held-out numbers",
     "the in-loop number must be cited with its population (seen tasks vs all); held-out is two tasks."),
    ("C2. Failure alarms come early enough to act on",
     "A CUSUM on the policy's own plan acceleration, calibrated at a 10% false-alarm target (realised {p2.cal.false_alarm}), alarms before timeout in {p2.cal.catch_rate} of failed calibration episodes, "
     "at a median step {p2.cal.first_alarm_step} with a median {p2.cal.lead_budget} steps of budget left ({p2.cal.lead_ge150} of alarms leave at least 150).",
     "descriptive (the false-alarm rate is a pre-registered calibration target; catch rate and lead time are reported, not tested)", "200 calibration episodes, one policy."),
    ("C3. Second Look makes failures visible, at a cost in success",
     "On seen tasks the loop (pause, camera check, flag if changed, else retry) cuts silent failures from {p2.seen.P0.SF} (P0) to {p2.seen.P3.SF} (P3), with {p2.seen.P3.FP} of its flags on episodes that P0 failed, "
     "but success changes by {p2.H1|ci} (H1 not supported: a loss, not a gain). It beats random-timed alarms by {p2.H3} (H3 supported), and it does not beat blind retry on success (H2: silent failures {p2.H2.SF}, success {p2.H2.S}, not supported).",
     "confirmatory (H1, H2, H3 pre-registered, Holm-corrected; H1 and H2 not supported, H3 supported); silent-failure and flag-precision numbers descriptive",
     "the gain in visibility is not a gain in success; an alarm's scene change is often the robot's own progress."),
    ("C4. A grace on the first 'changed' verdict repairs part of the cost, not all",
     "Letting the first CHANGED verdict pass (PA) raises success over P3 by {p3.H1|ci} (H1 supported) and keeps silent failures at {p3.seen.PA.SF} against {p3.seen.P0.SF} for P0, "
     "but it misses two pre-registered bars: the silent-failure reduction is {p3.H2|ci}, whose interval reaches below the 10 pp margin (H2 not supported), and success against P0 is {p3.H3|ci} (non-inferiority margin -3 pp, H3 not supported).",
     "confirmatory (Part 3 hypotheses on seen tasks, pre-registered)", "five TEST seeds on init states already used by Parts 1-3; held-out tasks are two tasks."),
    ("C5. A learned retry-or-continue model adds little beyond 'retry'",
     "Choosing at the first alarm between retry and continue with a value model raises success over doing nothing by {p4.H1|ci} (H1 supported), but not over always-retry ({p4.H2|ci}) or over a task-id lookup ({p4.H3|ci}); "
     "it retries {p4.SH.retry_rate} of episodes against {p4.P2r.retry_rate} for always-retry with {p4.SH.rescue} rescues and {p4.SH.harm} harms (always-retry: {p4.P2r.rescue} and {p4.P2r.harm}). "
     "Adding a flag (SH+F) cuts silent failures to {p4.SH+F.SF} at {p4.SH+F.FP} flag precision and no measurable success cost ({p4.SH+F.S}).",
     "confirmatory (H1-H3 pre-registered, Holm; H1 supported, H2 and H3 not); the SH+F numbers descriptive", "TEST init states were already used by Parts 1-3 (sealed by new noise seeds only); the gain over P0 is not specific to the model."),
    ("C6. A noise-steering policy would have room, but not on every failure",
     "On alarmed failures one random retry rescues {p5.0.mean_p} of states and the best of 64 noise draws {p5.0.any64}: a gap of {p5.0.gap|ci} (GO), worth up to {p5.0.ceiling_episode} of TEST success over always-retry; "
     "yet {p5.0.dist_zero} of the states never succeed in 64 draws (habits on the policy's own manifold) and only {p5.0.dist_high} succeed usually.",
     "confirmatory (P5.0 GO rule pre-registered); the distribution of per-state rates descriptive", "40 states in 35 clusters; the pre-registered sizes were cut to fit the compute budget."),
    ("C7. The room is invisible: no critic and no feature sees which noise is good",
     "Within a state, a critic on the observation, the noise and the plan ranks successful draws above failed ones with AUROC {p5.1a.auc|ci} and best-of-8 gains only {p5.1a.k8_gain|ci} over a random retry "
     "(bars: 0.65 and 10 pp; NO-GO); single features reach {p5.1a.single.plan_accel}, {p5.1a.single.distance_to_medoid} and {p5.1a.single.noise_norm}. The first decision's noise carries {p5.1b.effect|ci} of the outcome (mixed), concentrated in a few states. "
     "A post hoc permutation null with the actual primary model gives p = {p5.1.posthoc.p}, which does not change the verdict.",
     "confirmatory (P5.1a and P5.1b rules pre-registered); single features and leave-one-task-out descriptive; the permutation p with the actual model post hoc",
     "no signal at this data size, not proof of absence; one linear critic family; P5.1b covers 10 mixed states."),
]


def expand(text):
    def sub(m):
        i, ci = m.group(1), bool(m.group(2))
        if i not in LED and i.startswith("p5.1.posthoc"):
            return "PENDING"
        if i not in LED:
            raise KeyError(f"claim cites unknown ledger row {i}")
        return LED[i]["display"] if ci else LED[i]["display"].split(" [")[0]
    return re.sub(r"\{([\w.+-]+)(\|ci)?\}", sub, text)


def main():
    L = ["# Claims of the paper", "", "Generated by `scripts/paper/build_claims.py`; every number is expanded from `paper/numbers.json` (the ledger). Ledger rows backing each claim are listed under it with their status.", ""]
    for title, text, status, caveat in CLAIMS:
        ids = sorted(set(re.findall(r"\{([\w.+-]+)(?:\|ci)?\}", text)))
        L += [f"## {title}", "", expand(text), "", f"- **Status:** {status}", f"- **Caveat:** {caveat}", "- **Ledger rows:** " + ", ".join(f"`{i}` ({LED[i]['status'] if i in LED else 'post hoc'})" for i in ids), ""]
    (PAPER / "claims.md").write_text("\n".join(L))
    print(len(CLAIMS), "claims")


if __name__ == "__main__":
    main()
