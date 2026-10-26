"""G1 determinism, G2 branching, G4 harmless extra renders, G5 budget inside the loop. Live: SmolVLA + a LIBERO env (+ the render Scene), ~5 GB RAM, ~1.3 GB VRAM.
  P2_LIVE=1 scripts/safe_run.sh python -m pytest tests/p2/test_live_loop.py -v
"""
import json
import time

import numpy as np
import pytest

import second_look.loop as L
from second_look.cusum import CusumParams
from second_look.loop import ArmSpec, RunCfg, run_arm_fresh, run_episode

pytestmark = pytest.mark.live
THR = {(0, "F1-max"): 1.764, (0, "recall>=0.99"): -2.104}
NEVER = CusumParams.make(0.0, 1.0, 1e12)                 # no natural alarm: only forced ones
SCALE = np.ones(70, np.float32)


@pytest.fixture(scope="module")
def rig():
    r = L.Rig()
    yield r
    r.close()


def cfg_for(task, init, force=(), **kw):
    return RunCfg("test", task, init, f"g-{task}-{init}", NEVER, SCALE, thresholds=THR, force_alarms=frozenset(force), **kw)


def core(res):
    """Everything that must be bitwise equal between two runs of the same thing (no wall-clock, no frames)."""
    return {"trace": res["log"]["trace"], "decisions": res["log"]["decisions"], "events": res["log"]["events"],
            **{k: res[k] for k in ("success", "flagged", "flag_reason", "elapsed", "executed", "paused", "returned", "rewinds", "invalid_rewinds", "decisions")}}


def same(a, b):
    return json.dumps(core(a), sort_keys=True, default=str) == json.dumps(core(b), sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------- G1
@pytest.mark.parametrize("task,init", [(0, 20), (3, 21), (7, 22)])
def test_g1_determinism(rig, task, init):
    a = run_episode(rig, cfg_for(task, init), [])["P0"]
    b = run_episode(rig, cfg_for(task, init), [])["P0"]
    assert len(a["log"]["trace"]) > 40
    assert a["log"]["trace"] == b["log"]["trace"], "actions / states differ between two runs"
    assert a["log"]["decisions"] == b["log"]["decisions"], "accel scores or observations differ"
    assert same(a, b)


def test_g1_history_independence(rig):
    """An episode must not depend on what ran before it in the process (regression: a late-destroyed render context of an earlier env turned the second
    episode's camera images into striped garbage; second_look.glfix)."""
    a = run_episode(rig, cfg_for(3, 21), [])["P0"]
    run_episode(rig, cfg_for(0, 40), [])
    run_episode(rig, cfg_for(5, 41), [])
    c = run_episode(rig, cfg_for(3, 21), [])["P0"]
    assert [x["obs"] for x in a["log"]["decisions"]] == [x["obs"] for x in c["log"]["decisions"]], "policy observations depend on the history of the process"
    assert same(a, c)


# ---------------------------------------------------------------------------------------------- G2
G2_EPISODES = [(0, 20, 6, "P2"), (1, 21, 8, "P3"), (2, 22, 5, "P4"), (3, 23, 7, "P6"), (4, 24, 9, "P1"),
               (5, 25, 6, "P2"), (7, 26, 12, "P3"), (9, 27, 4, "P4"), (6, 28, 10, "P2"), (8, 29, 8, "P6")]
G2_OUTCOMES: list = []


@pytest.mark.parametrize("task,init,d,kind", G2_EPISODES)
def test_g2_branch_equals_fresh_replay(rig, task, init, d, kind):
    arm = ArmSpec(f"{kind}-t", kind, "accel", target="L2", N=2, op="F1-max")
    n = len(run_episode(rig, cfg_for(task, init), [])["P0"]["log"]["decisions"])          # the baseline must still be running at the branch decision
    d = min(d, n - 2)
    cfg = cfg_for(task, init, force=(d,))
    res = run_episode(rig, cfg, [arm])
    branched, base = res[arm.name], res["P0"]
    fresh = run_arm_fresh(rig, cfg, arm, d)
    assert len(branched["log"]["trace"]) > 10 * d
    assert same(branched, fresh), f"branch at decision {d} ({kind}) differs from the fresh replay"
    G2_OUTCOMES.append({"task": task, "init": init, "d": d, "kind": kind, "baseline_success": base["success"], "arm_success": branched["success"], "arm_flagged": branched["flagged"],
                        "rewinds": branched["rewinds"], "events": [e["kind"] for e in branched["log"]["events"]]})


def test_g2_summary_includes_failures():
    if len(G2_OUTCOMES) < len(G2_EPISODES):
        pytest.skip("run the whole module")
    open("results/part2/logs/g2_outcomes.json", "w").write(json.dumps(G2_OUTCOMES, indent=1))
    assert any(not o["baseline_success"] for o in G2_OUTCOMES), "no failing baseline episode among the 10"


# ---------------------------------------------------------------------------------------------- G4
def test_g4_extra_renders_are_harmless(rig):
    a = run_episode(rig, cfg_for(1, 30), [])["P0"]
    b = run_episode(rig, cfg_for(1, 30, render_every_decision=True), [])["P0"]
    assert a["log"]["trace"] == b["log"]["trace"], "actions or states changed because we rendered extra frames"
    assert [x["obs"] for x in a["log"]["decisions"]] == [x["obs"] for x in b["log"]["decisions"]], "policy observations changed"
    assert same(a, b)


# ---------------------------------------------------------------------------------------------- G5 inside the loop
def test_g5_two_rewinds_clock(rig):
    """A quiet scene (alarm at decision 3, before the robot touches anything), target L2 = decision 1."""
    arm = ArmSpec("P2-g5", "P2", "accel", target="L2", N=2)
    res = run_episode(rig, cfg_for(2, 31, force=(3,)), [arm])["P2-g5"]
    assert res["rewinds"] == 2 and res["flag_reason"] == "retry_limit" and res["flagged"], res
    # d=3 is reached 3 times: 30 steps, then 2 x (replay decisions 1,2 = 20 steps) executed; 2 pauses of 10; 2 returns of 2 decisions (20 steps each)
    assert (res["executed"], res["paused"], res["returned"], res["elapsed"]) == (70, 20, 40, 130), res


def test_g5_success_during_a_pause_counts(rig, monkeypatch):
    real, n = L.env_step, [0]

    def fake(rig_, cfg, st, action, tag):
        ok = real(rig_, cfg, st, action, tag)
        if tag == "pause":
            n[0] += 1
            return n[0] == 3
        return ok
    monkeypatch.setattr(L, "env_step", fake)
    res = run_episode(rig, cfg_for(2, 32, force=(4,)), [ArmSpec("P2-s", "P2", "accel", N=1)])["P2-s"]
    assert res["success"] and res["paused"] == 3 and res["elapsed"] == 40 + 3 and not res["flagged"], res


def test_g5_budget_cuts_a_retry_short(rig):
    """A late alarm: the return cost eats the remaining budget, the episode ends as a timeout (not a flag)."""
    res = run_episode(rig, cfg_for(9, 27, force=(28,)), [ArmSpec("P2-late", "P2", "accel", target="L10", N=3)])["P2-late"]          # (9, 27): a baseline that runs the whole budget
    assert res["cut_short"] and res["elapsed"] == 300 and res["returned"] == 100 and not res["flagged"] and not res["success"], res
