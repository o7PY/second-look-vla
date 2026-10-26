"""Memoisation of the first pause / settled frame / DeltaDINO score across arms branched from one alarm must change nothing: 5 episodes, 13 arms of every kind, run with
`memoise=True` and `memoise=False`; every trace hash, decision log, event (gate scores included), outcome and gate frame must be bitwise equal. Also times both.
  P2_LIVE=1 scripts/safe_run.sh python -m pytest tests/p2/test_live_memo.py -v -s
"""
import json
import time
from dataclasses import replace

import numpy as np
import pytest

import second_look.loop as L
from second_look.cusum import CusumParams
from second_look.loop import ArmSpec, RunCfg, run_episode

pytestmark = pytest.mark.live
THR = {(0, "F1-max"): 1.764, (0, "recall>=0.99"): -2.104}
NEVER = CusumParams.make(0.0, 1.0, 1e12)
ARMS = [ArmSpec("P1", "P1"), ArmSpec("P2-L2-N2", "P2", target="L2", N=2), ArmSpec("P2-L4-N1", "P2", target="L4", N=1),
        ArmSpec("P3-L2-N3", "P3", target="L2", N=3, op="F1-max"), ArmSpec("P3-L2-N1-r99", "P3", target="L2", N=1, op="recall>=0.99"),
        ArmSpec("P3-L1-N2", "P3", target="L1", N=2, op="F1-max"), ArmSpec("P3-LC4-N2", "P3", target="LC4", N=2, op="F1-max"),
        ArmSpec("P3-L6-N2", "P3", target="L6", N=2, op="F1-max"), ArmSpec("P4-L2-N2", "P4", target="L2", N=2), ArmSpec("P4-L10-N1", "P4", target="L10", N=1),
        ArmSpec("P6-L2", "P6", target="L2", op="F1-max"), ArmSpec("P3tele-L4", "P3", target="L4", N=2, op="F1-max", teleport=True),
        ArmSpec("P5-P3-L2", "P3", trigger="p5", target="L2", N=2, op="F1-max")]
EPISODES = [(0, 20), (1, 21), (3, 22), (5, 23), (9, 24)]


@pytest.fixture(scope="module")
def rig():
    r = L.Rig()
    yield r
    r.close()


def cfg_for(task, init, memoise):
    return RunCfg("memo", task, init, f"m-{task}-{init}", NEVER, np.ones(70, np.float32), thresholds=THR, force_alarms=frozenset({4, 6}), p5_hazard=np.full(30, 0.25),
                  memoise=memoise)


def core(r):
    return json.dumps({"trace": r["log"]["trace"], "decisions": r["log"]["decisions"], "events": r["log"]["events"],
                       **{k: r[k] for k in ("success", "flagged", "flag_reason", "elapsed", "executed", "paused", "returned", "rewinds", "invalid_rewinds", "decisions", "cut_short")}},
                      sort_keys=True, default=str)


TIMES = {"memo": 0.0, "plain": 0.0}


@pytest.mark.parametrize("task,init", EPISODES)
def test_memoised_equals_unmemoised(rig, task, init):
    t0 = time.time(); a = run_episode(rig, cfg_for(task, init, True), ARMS); TIMES["memo"] += time.time() - t0
    t0 = time.time(); b = run_episode(rig, cfg_for(task, init, False), ARMS); TIMES["plain"] += time.time() - t0
    assert set(a) == set(b) and len(a) == 1 + len(ARMS) - sum(1 for k in a if False)
    n_gate = 0
    for name in a:
        assert core(a[name]) == core(b[name]), f"{name}: memoised run differs from the unmemoised one"
        assert len(a[name]["pairs"]) == len(b[name]["pairs"])
        for (r1, s1), (r2, s2) in zip(a[name]["pairs"], b[name]["pairs"]):
            assert np.array_equal(r1, r2) and np.array_equal(s1, s2), f"{name}: gate frames differ"
        n_gate += sum(e["kind"] == "gate" for e in a[name]["log"]["events"])
    assert n_gate >= 8, "the test must exercise the gate"


def test_report_times():
    if TIMES["plain"] == 0:
        pytest.skip("run the whole module")
    print(f"\nwall time for 5 episodes x {len(ARMS)} arms: memoised {TIMES['memo']:.0f} s, unmemoised {TIMES['plain']:.0f} s, saved {100 * (1 - TIMES['memo'] / TIMES['plain']):.0f}%")
    open("results/part2/logs/memo_times.json", "w").write(json.dumps(TIMES))
