import subprocess

import pytest

from second_look import p4guard as G


def git(repo, *a):
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q"); git(tmp_path, "config", "user.email", "t@t"); git(tmp_path, "config", "user.name", "t")
    for d in ("results/part4/notes", "configs"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "x").write_text("x"); git(tmp_path, "add", "x"); git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.mark.parametrize("which,files,what", [("check_gen", G.GEN, "GEN"), ("check_dev", G.DEV, "DEV"), ("check_test", G.TEST, "TEST")])
def test_every_required_file_must_be_tracked_and_clean(repo, which, files, what):
    fn = getattr(G, which)
    for rel in files:
        with pytest.raises(PermissionError, match=f"{what} refused: {rel} does not exist"):
            fn(repo)
        (repo / rel).write_text("v1\n")
        with pytest.raises(PermissionError, match=f"{what} refused: {rel} is not tracked"):
            fn(repo)
        git(repo, "add", rel)
        with pytest.raises(PermissionError, match=f"{what} refused: {rel} has uncommitted"):
            fn(repo)
        git(repo, "commit", "-qm", f"add {rel}")
    fn(repo)
    (repo / files[0]).write_text("edited after the commit\n")
    with pytest.raises(PermissionError, match=f"{what} refused: {files[0]} has uncommitted"):
        fn(repo)


def test_the_stages_are_nested():
    assert set(G.GEN) < set(G.DEV) < set(G.TEST) and G.MODEL in G.DEV and G.DECISION in G.TEST and G.DECISION not in G.DEV
