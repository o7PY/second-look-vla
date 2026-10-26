import subprocess

import pytest

from second_look import p2guard as G


def git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q"); git(tmp_path, "config", "user.email", "t@t"); git(tmp_path, "config", "user.name", "t")
    (tmp_path / "results/part2/notes").mkdir(parents=True); (tmp_path / "configs").mkdir()
    (tmp_path / "x").write_text("x"); git(tmp_path, "add", "x"); git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_dev_needs_the_plan_in_history_and_the_frozen_config(repo):
    with pytest.raises(PermissionError, match="does not exist"):
        G.check_dev(repo)
    p = repo / G.PLAN
    p.write_text("plan v1\n")
    with pytest.raises(PermissionError, match="never committed"):
        G.check_dev(repo)
    git(repo, "add", "-f", G.PLAN); git(repo, "commit", "-qm", "plan")
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_frozen.yaml does not exist"):          # plan fine, frozen config missing
        G.check_dev(repo)
    (repo / G.FROZEN).write_text("accel: {}\n")
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_frozen.yaml is not tracked"):
        G.check_dev(repo)
    git(repo, "add", G.FROZEN)
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_frozen.yaml has uncommitted"):
        G.check_dev(repo)
    git(repo, "commit", "-qm", "freeze")
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_gate_cal.yaml does not exist"):         # frozen config fine, in-loop gate thresholds missing
        G.check_dev(repo)
    (repo / G.GATE_CAL).write_text("thresholds: {}\n")
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_gate_cal.yaml is not tracked"):
        G.check_dev(repo)
    git(repo, "add", G.GATE_CAL)
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_gate_cal.yaml has uncommitted"):
        G.check_dev(repo)
    git(repo, "commit", "-qm", "gate cal")
    G.check_dev(repo)
    (repo / G.GATE_CAL).write_text("thresholds: {edited: 1}\n")                                                  # edited after the commit
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_gate_cal.yaml has uncommitted"):
        G.check_dev(repo)
    git(repo, "checkout", "--", G.GATE_CAL)
    (repo / G.FROZEN).write_text("accel: {changed: 1}\n")                                                       # edited after the commit
    with pytest.raises(PermissionError, match="DEV refused: configs/part2_frozen.yaml has uncommitted"):
        G.check_dev(repo)
    git(repo, "checkout", "--", G.FROZEN)
    git(repo, "rm", "--cached", "-q", G.PLAN); git(repo, "commit", "-qm", "untrack")             # like the real repo after the .md untracking
    G.check_dev(repo)                                                                              # still fine: unchanged since its commit
    p.write_text("plan v2 (edited after the fact)\n")
    with pytest.raises(PermissionError, match="differs from its last committed"):
        G.check_dev(repo)


def test_test_needs_committed_decision_and_frozen_config(repo):
    (repo / G.PLAN).write_text("plan\n"); git(repo, "add", "-f", G.PLAN); git(repo, "commit", "-qm", "plan")
    with pytest.raises(PermissionError, match="p2_5_dev_decision.json does not exist"):
        G.check_test(repo)
    for rel in (G.DECISION, G.FROZEN, G.GATE_CAL):
        (repo / rel).write_text("{}\n")
    with pytest.raises(PermissionError, match="not tracked"):
        G.check_test(repo)
    git(repo, "add", G.DECISION, G.FROZEN, G.GATE_CAL)
    with pytest.raises(PermissionError, match="uncommitted"):
        G.check_test(repo)
    git(repo, "commit", "-qm", "decision")
    G.check_test(repo)
    (repo / G.FROZEN).write_text("{changed: 1}\n")
    with pytest.raises(PermissionError, match="uncommitted"):
        G.check_test(repo)
    git(repo, "checkout", "--", G.FROZEN)
    G.check_test(repo)
    (repo / G.GATE_CAL).write_text("{changed: 1}\n")
    with pytest.raises(PermissionError, match="part2_gate_cal.yaml has uncommitted"):
        G.check_test(repo)


@pytest.mark.skip(reason="compares against commit 4511dbf of the ORIGINAL repo; this repo has fresh history (see PROVENANCE.md)")
def test_real_repo_plan_is_the_committed_one():
    """The real repo: the plan on disk is exactly what was committed in 4511dbf (before any DEV run)."""
    ok, msg = G.committed_in_history(G.PLAN)
    assert ok and msg == "4511dbf", msg


def test_real_repo_dev_needs_the_frozen_config_and_gate_cal_until_committed():
    ok = G.tracked_and_clean(G.FROZEN)[0] and G.tracked_and_clean(G.GATE_CAL)[0]
    if ok:
        assert G.check_dev()                    # after P2.3
    else:
        with pytest.raises(PermissionError, match="DEV refused"):
            G.check_dev()
