"""Sealed test set guard (SPEC_before_part2.md R5/R7).

``test_policy2`` may be used by NO code that loads data until the model choice is written down and committed: `results/part1/b2/notes/r6_decision.md` must exist,
be tracked by git and have no uncommitted changes. The guard sits in the single entry points that read split data (`PairDataset`, `build_cache`),
so evaluation, review sheets and cache building all go through it. (Code that opens the shards with np.load directly is not covered; the only
script that does so is the pair builder, which prints counts only.)"""

from __future__ import annotations

import subprocess
from pathlib import Path

from second_look.scene import ROOT

SEALED = frozenset({"test_policy2"})
DECISION = ROOT / "results" / "part1" / "b2" / "notes" / "r6_decision.md"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def decision_committed(decision: Path = DECISION, repo: Path = ROOT) -> tuple[bool, str]:
    if not decision.exists():
        return False, f"{decision} does not exist"
    rel = str(decision.resolve().relative_to(repo.resolve()))
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        return False, f"{rel} is not tracked by git (commit it first)"
    if _git(repo, "status", "--porcelain", "--", rel).stdout.strip():
        return False, f"{rel} has uncommitted changes (commit it first)"
    return True, "ok"


def check_sealed(split: str, decision: Path = DECISION, repo: Path = ROOT) -> None:
    """Raise PermissionError if ``split`` is sealed and the decision file is not committed."""
    if split not in SEALED:
        return
    ok, why = decision_committed(decision, repo)
    if not ok:
        raise PermissionError(f"'{split}' is sealed until the model choice is committed: {why}. See SPEC_before_part2.md R5-R7.")
