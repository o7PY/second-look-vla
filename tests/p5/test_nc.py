"""P5.0 tests. Non-live: the p5nc namespace changes no Part 1-4 key, the guard. Live (P2_LIVE=1, ~5 GB):
  P2_LIVE=1 scripts/safe_run.sh python -m pytest tests/p5 -v
Live conditions: (1) the replay of two GEN states equals the GEN log up to the branch decision and in outcome; (2) the GEN controls (R1-R4, C1-C4 with GEN's own keys) reproduce the logged outcomes;
(3) a p5nc retry and a p5nc continue branch run twice are bitwise equal; (4) mode C differs from P0 at the alarm decision (re-plan) and executes no pause."""
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from second_look import p5guard, seeding

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def test_old_keys_unchanged():
    # values computed with src/second_look/seeding.py of commit part4-complete
    assert seeding.noise_seed("p4gen", 3, 7, "p4gen-1", 0, 12, 0) == int.from_bytes(__import__("hashlib").sha256(b"p4gen|3|7|p4gen-1|0|12|0").digest()[:8], "little") & ((1 << 63) - 1)
    assert seeding.noise_seed("p4gen", 3, 7, "p4gen-1", "reroll", 12, 1) == int.from_bytes(__import__("hashlib").sha256(b"p4gen|3|7|p4gen-1|reroll|12|1").digest()[:8], "little") & ((1 << 63) - 1)
    for bad in ("p5nc", "p5nc-1", "p5ncx", "other", -1, True, 1.5):
        with pytest.raises(ValueError):
            seeding.noise_seed("p4gen", 0, 0, "s", bad, 0, 0)


def test_p5nc_keys_are_new():
    seen = {seeding.noise_seed("p4gen", 3, 7, "p4gen-1", iv, 12, 0) for iv in list(range(0, 40)) + [2001, 2002, 2003, 2004, "reroll"]}
    new = {seeding.noise_seed("p4gen", 3, 7, "p4gen-1", f"p5nc{j}", 12, 0) for j in range(64)}
    assert len(new) == 64 and not (new & seen)


def test_guard_refuses_test_inits():
    p5guard.check_inits([{"task": 0, "init": 0}, {"task": 1, "init": 19}, {"task": 1, "init": 40}])
    for init in (20, 39, 50):
        with pytest.raises(PermissionError):
            p5guard.check_inits([{"task": 0, "init": init}])


def test_guard_refuses_uncommitted_plan(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    with pytest.raises(PermissionError):
        p5guard.check_run(tmp_path)


def test_states_file_has_only_gen_inits():
    p = ROOT / "results/part5/nc/states.json"
    if not p.exists():
        pytest.skip("states.json not built")
    st = json.load(open(p))["states"]
    p5guard.check_inits(st)
    assert all(0 <= s["init"] < 20 for s in st)
    assert sum(s["pop"] == "A" for s in st) <= 60 and all(not s["p0_success"] for s in st if s["pop"] == "A") and all(s["p0_success"] for s in st if s["pop"] == "C")


# ------------------------------------------------------------------------------------------------ live
live = pytest.mark.live


@pytest.fixture(scope="module")
def rig():
    import second_look.loop as L
    r = L.Rig()
    yield r
    r.close()


def _conf():
    import p2_run
    conf = p2_run.load_config("configs/part2_frozen.yaml")
    conf["gate_cal"] = p2_run.load_config("configs/part2_gate_cal.yaml")
    return conf


def _state(pop, i, control=False):
    st = [s for s in json.load(open(ROOT / "results/part5/nc/states.json"))["states"] if s["pop"] == pop]
    s = dict(st[i])
    s["control"] = control
    return s


@live
@pytest.mark.parametrize("which", [("A", 0), ("A", 30)])
def test_replay_and_controls_reproduce_gen(rig, which):
    import p4_common as C
    import p5_nc_run as R
    s = _state(*which, control=True)
    recs = C.load("results/part4/gen")
    rec = R.run_state(rig, _conf(), s, 2, 1, 1000, recs[(s["task"], s["init"], s["seed_id"])])
    assert rec["replay_equal"], "replay of the P0 episode differs from the GEN log"
    assert rec["d_equal"], "the branch decision differs from the logged alarm decision"
    assert all(c["equal"] for c in rec["controls"].values()), rec["controls"]


@live
def test_p5nc_branch_twice_bitwise_equal_and_mode_c_replans(rig):
    import p4_common as C
    import p5_nc_run as R
    s = _state("A", 5)
    recs = C.load("results/part4/gen")
    a = R.run_state(rig, _conf(), s, 2, 2, 1000, recs[(s["task"], s["init"], s["seed_id"])])
    b = R.run_state(rig, _conf(), s, 2, 2, 1000, recs[(s["task"], s["init"], s["seed_id"])])
    for k in a["branches"]:
        x, y = dict(a["branches"][k]), dict(b["branches"][k])
        assert x == y, k
    assert a["branches"]["R0"]["p5"]["chunk"] != a["branches"]["R1"]["p5"]["chunk"]
    assert a["branches"]["C0"]["p5"]["chunk"] != a["branches"]["C1"]["p5"]["chunk"]
    assert a["branches"]["C0"]["return_cost"] is None and a["branches"]["C0"]["rewinds"] == 0          # no pause, no rewind
    assert a["branches"]["R0"]["rewinds"] == 1
    assert a["branches"]["R0"]["p5"]["noise_shape"] == [1, 50, 32] and "end_label" in a["branches"]["R0"]["p5"]
