"""R5: the sealed test set cannot be read until the model-choice note is committed."""
import subprocess

import pytest

from second_look import sealed


def _git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q"); _git(tmp_path, "config", "user.email", "t@t"); _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "results" / "part1" / "b2" / "notes").mkdir(parents=True)
    (tmp_path / "README").write_text("x"); _git(tmp_path, "add", "README"); _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_other_splits_are_never_blocked(repo):
    for sp in ("train", "val", "test", "test_heldout_tasks", "test_policy", "probe_static"):
        sealed.check_sealed(sp, repo / "results/part1/b2/notes/r6_decision.md", repo)


def test_refused_without_the_note(repo):
    with pytest.raises(PermissionError, match="does not exist"):
        sealed.check_sealed("test_policy2", repo / "results/part1/b2/notes/r6_decision.md", repo)


def test_refused_when_untracked_or_modified_allowed_when_committed(repo):
    d = repo / "results/part1/b2/notes/r6_decision.md"
    d.write_text("choice: A\n")
    with pytest.raises(PermissionError, match="not tracked"):
        sealed.check_sealed("test_policy2", d, repo)
    _git(repo, "add", str(d.relative_to(repo)))
    with pytest.raises(PermissionError, match="uncommitted"):                       # staged but not committed
        sealed.check_sealed("test_policy2", d, repo)
    _git(repo, "commit", "-qm", "R6 decision")
    sealed.check_sealed("test_policy2", d, repo)                                    # now allowed
    d.write_text("choice: C\n")                                                     # edited after the commit
    with pytest.raises(PermissionError, match="uncommitted"):
        sealed.check_sealed("test_policy2", d, repo)


def test_dataset_entry_points_enforce_the_guard(monkeypatch, tmp_path):
    from second_look import dataset
    monkeypatch.setattr(sealed, "DECISION", tmp_path / "missing.md")
    monkeypatch.setattr(dataset, "check_sealed", lambda split: sealed.check_sealed(split, tmp_path / "missing.md"))
    with pytest.raises(PermissionError):
        dataset.PairDataset("test_policy2")
    with pytest.raises(PermissionError):
        dataset.build_cache("test_policy2")
