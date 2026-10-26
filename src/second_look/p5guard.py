"""Guards of Part 5 / P5.0 (results/part5/notes/p5_0_plan.txt): the noise-ceiling test uses ONLY states of the Part 4 GEN split (init 0-19; DEV init 40-49 are also allowed ranges) and never a TEST init state (20-39);
a non-bench run needs the plan and the yaml (final K, state count, cuts) committed and clean, with the frozen Part 2 configs."""
from __future__ import annotations

from pathlib import Path

from second_look.p2guard import FROZEN, GATE_CAL, tracked_and_clean
from second_look.scene import ROOT

PLAN = "results/part5/notes/p5_0_plan.txt"
YAML = "configs/part5_noise_ceiling.yaml"
ALLOWED_INITS = frozenset(range(0, 20)) | frozenset(range(40, 50))
RUN = (PLAN, YAML, FROZEN, GATE_CAL)


def check_inits(states) -> None:
    bad = [(s["task"], s["init"]) for s in states if int(s["init"]) not in ALLOWED_INITS]
    if bad:
        raise PermissionError(f"P5.0 refused: init states outside the GEN / DEV ranges (0-19, 40-49): {bad[:5]}")


def check_run(repo: Path = ROOT) -> None:
    for rel in RUN:
        ok, msg = tracked_and_clean(rel, repo)
        if not ok:
            raise PermissionError(f"P5.0 run refused: {msg}")


PLAN1 = "results/part5/notes/p5_1_plan.txt"
YAML1 = "configs/part5_critic.yaml"
RUN1 = (PLAN1, YAML1, FROZEN, GATE_CAL)


def check_p51(repo: Path = ROOT) -> None:
    """P5.1: the plan and the parameters are committed and clean before any critic fit and any P5.1b branch."""
    for rel in RUN1:
        ok, msg = tracked_and_clean(rel, repo)
        if not ok:
            raise PermissionError(f"P5.1 refused: {msg}")
