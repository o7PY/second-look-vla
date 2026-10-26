"""Git-order guards of Part 4 (SPEC_part4.md section 10, exit criteria Z2-Z4). `*.md` is untracked, so the records that must exist BEFORE a run are non-.md files:
  * GEN needs `results/part4/notes/p4_analysis_plan.txt` and `configs/part4_arms.yaml` (the pre-registration, P4.3) and the frozen Part 2 detector config and gate thresholds tracked and clean (Z2);
  * DEV needs, in addition, the frozen model `configs/part4_model.pkl` and its record `configs/part4_model.json` (sha256, feature code version) tracked and clean (Z3);
  * TEST needs, in addition, the DEV decision `results/part4/notes/p4_dev_decision.json` tracked and clean (Z4).
"""
from __future__ import annotations

from pathlib import Path

from second_look.p2guard import FROZEN, GATE_CAL, tracked_and_clean
from second_look.scene import ROOT

PLAN = "results/part4/notes/p4_analysis_plan.txt"
ARMS = "configs/part4_arms.yaml"
MODEL = "configs/part4_model.pkl"
MODEL_JSON = "configs/part4_model.json"
DECISION = "results/part4/notes/p4_dev_decision.json"
GEN = (PLAN, ARMS, FROZEN, GATE_CAL)
DEV = GEN + (MODEL, MODEL_JSON)
TEST = DEV + (DECISION,)


def _check(what: str, files, repo: Path) -> None:
    for rel in files:
        ok, msg = tracked_and_clean(rel, repo)
        if not ok:
            raise PermissionError(f"{what} refused: {msg}")


def check_gen(repo: Path = ROOT) -> None:
    _check("GEN", GEN, repo)


def check_dev(repo: Path = ROOT) -> None:
    _check("DEV", DEV, repo)


def check_test(repo: Path = ROOT) -> None:
    _check("TEST", TEST, repo)
