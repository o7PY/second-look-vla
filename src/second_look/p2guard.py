"""Git-order guards of Part 2 (SPEC_part2.md §8, Z5, Z6).

`.md` files are not tracked any more (2026-10-10), so a record that must be committed BEFORE a run is a non-.md file:
  * DEV may start only if the pre-registered analysis plan `results/part2/notes/p2_4_analysis_plan.md` exists in git HISTORY and its content on disk equals the last committed
    version (the file itself is untracked now; it was committed in 4511dbf, before any DEV run), AND `configs/part2_frozen.yaml` (P2.3) and `configs/part2_gate_cal.yaml` (in-loop gate thresholds) are tracked and clean;
  * TEST may start only if `results/part2/notes/p2_5_dev_decision.json` (written together with the .md note of the DEV choice) is tracked, has no uncommitted changes, and
    `configs/part2_frozen.yaml` and `configs/part2_gate_cal.yaml` are tracked and clean (the frozen detectors / thresholds / chosen config).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from second_look.scene import ROOT

PLAN = "results/part2/notes/p2_4_analysis_plan.md"
DECISION = "results/part2/notes/p2_5_dev_decision.json"
FROZEN = "configs/part2_frozen.yaml"
GATE_CAL = "configs/part2_gate_cal.yaml"


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=text)


def tracked_and_clean(rel: str, repo: Path = ROOT) -> tuple[bool, str]:
    if not (repo / rel).exists():
        return False, f"{rel} does not exist"
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        return False, f"{rel} is not tracked by git (commit it first; non-.md files only, .md is ignored)"
    if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
        return False, f"{rel} has uncommitted changes (commit it first)"
    return True, "ok"


def committed_in_history(rel: str, repo: Path = ROOT) -> tuple[bool, str]:
    """The file exists on disk and equals the last version that was ever committed."""
    if not (repo / rel).exists():
        return False, f"{rel} does not exist"
    h = _git(repo, "log", "-1", "--diff-filter=AM", "--format=%H", "--", rel).stdout.strip()
    if not h:
        return False, f"{rel} was never committed"
    blob = _git(repo, "show", f"{h}:{rel}", text=False).stdout
    if blob != (repo / rel).read_bytes():
        return False, f"{rel} differs from its last committed version {h[:7]}"
    return True, h[:7]


def check_dev(repo: Path = ROOT) -> str:
    ok, msg = committed_in_history(PLAN, repo)
    if not ok:
        raise PermissionError(f"DEV refused: {msg}")
    for rel in (FROZEN, GATE_CAL):                                  # the frozen calibration (P2.3) and the in-loop gate thresholds must be committed before any DEV episode
        ok2, msg2 = tracked_and_clean(rel, repo)
        if not ok2:
            raise PermissionError(f"DEV refused: {msg2}")
    return msg


def check_test(repo: Path = ROOT) -> None:
    for rel in (DECISION, FROZEN, GATE_CAL):
        ok, msg = tracked_and_clean(rel, repo)
        if not ok:
            raise PermissionError(f"TEST refused: {msg}")
    check_dev(repo)
