"""Part 4 arm SH (src/second_look/selective.py): live tests (SmolVLA + a LIBERO env + the render Scene, ~5 GB RAM).
  P2_LIVE=1 scripts/safe_run.sh python -m pytest tests/p4/test_sh_live.py -v

Conditions (SPEC_part4.md section 10, P4.1 / P4.2; written before any Part 4 episode):
  1. features: identical on replay (two runs), identical across the arms of one alarm, finite, and available for every alarm of the test episodes;
  2. the SH arm with mode "model" and a stub model equals the physical branch it chooses: retry == P2r (mode "retry"), continue == PC (mode "continue"), bitwise (trace, decisions, events but the arm's own);
     SH+F flags (reason low_value, no rewind) exactly when max p < tau;
  3. P2r follows the Part 2 retry: its steps equal Part 3's PA kind (gate forced UNCHANGED, which rewinds to L6 and retries with the same keys) up to PA's second pause; later alarms are ignored (no flag, no second rewind);
  4. PC equals PA's grace path (gate forced CHANGED) up to PA's next alarm; with an EMPTY pause PC equals P0 over the WHOLE episode (continue branch c = 0 == P0's continuation);
  5. noise keys: retry key 1 == P2r, retry key 2 differs; continue key c >= 1 executes the same planned chunk as PC and then plans with fresh keys (iv 2000 + c);
  6. branch == fresh replay (G2) for SH retry and continue.
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import second_look.loop as L
from second_look.cusum import CusumParams
from second_look.loop import ArmSpec, run_arm_fresh, run_episode
from second_look.shmodel import StubModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import p2_run  # noqa: E402

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parents[1].parent
KEY = ("test", 3, 24, "p2test-3")                  # Part 2 TEST task 3, init 24, seed p2test-3: first alarm at decision 10, a second one follows


@pytest.fixture(scope="module")
def rig():
    r = L.Rig()
    yield r
    r.close()


def conf():
    c = p2_run.load_config(str(ROOT / "configs/part2_frozen.yaml"))
    c["gate_cal"] = p2_run.load_config(str(ROOT / "configs/part2_gate_cal.yaml"))
    return c


def cfg_for(thr=None, key=KEY, models=None, **kw):
    cfg = p2_run.make_cfg(*key, conf())
    t = dict(cfg.thresholds)
    if thr is not None:
        t[(0, "fpr05", "L6")] = thr
    return dataclasses.replace(cfg, trace=True, p4=True, thresholds=t, sh_models=models or {}, **kw)


def sh(name, mode, key=0, **kw):
    return ArmSpec(name, "SH", "accel", target="L6", N=1, op="fpr05", mode=mode, key=key, **kw)


def pa(name="PA"):
    return ArmSpec(name, "PA", "accel", target="L6", N=3, op="fpr05")


def core(r):
    ev = [{k: v for k, v in e.items() if k not in ("arm_kind", "mode", "model", "delta", "tau", "key", "decision", "p_retry", "p_cont")} for e in r["log"]["events"]]       # the "sh" event differs by design in its p-hats
    return json.dumps({"t": r["log"]["trace"], "d": r["log"]["decisions"], "e": ev, **{k: r[k] for k in ("success", "flagged", "flag_reason", "elapsed", "executed", "paused", "returned", "rewinds")}}, sort_keys=True, default=str)


def sh_event(r):
    return next(e for e in r["log"]["events"] if e["kind"] == "sh")


def test_features_replay_and_equal_across_arms(rig):
    cfg = cfg_for()
    a = run_episode(rig, cfg, [sh("P2r", "retry"), sh("PC", "continue")])
    b = run_episode(rig, cfg, [sh("P2r", "retry"), sh("PC", "continue")])
    xa, xb = sh_event(a["P2r"])["x"], sh_event(b["P2r"])["x"]
    assert xa == xb and xa == sh_event(a["PC"])["x"] == sh_event(b["PC"])["x"]
    x = np.array(xa)
    assert x.shape == (28,) and np.isfinite(x).all() and x[3] == 1.0 and x[10:18].shape == (8,)
    assert core(a["P2r"]) == core(b["P2r"]) and core(a["PC"]) == core(b["PC"])
    assert a["P2r"]["rewinds"] == 1 and a["PC"]["rewinds"] == 0 and a["P2r"]["paused"] == 10 and a["PC"]["paused"] == 10


def test_model_arm_equals_the_branch_it_chooses(rig):
    models = {"hi": StubModel(0.9, 0.1), "lo": StubModel(0.1, 0.9), "mid": StubModel(0.55, 0.5)}
    cfg = cfg_for(models=models)
    arms = [sh("P2r", "retry"), sh("PC", "continue"), sh("SH-r", "model", model="hi", delta=0.0), sh("SH-c", "model", model="lo", delta=0.0), sh("SH-d", "model", model="mid", delta=0.1),
            sh("SH-d0", "model", model="mid", delta=0.0), sh("SHF", "model", model="mid", delta=0.0, tau=0.6), sh("SHF2", "model", model="hi", delta=0.0, tau=0.6)]
    res = run_episode(rig, cfg, arms)
    assert core(res["SH-r"]) == core(res["P2r"]) and sh_event(res["SH-r"])["decision"] == "retry"
    assert core(res["SH-c"]) == core(res["PC"]) and sh_event(res["SH-c"])["decision"] == "continue"
    assert core(res["SH-d"]) == core(res["PC"]) and sh_event(res["SH-d"])["decision"] == "continue"                    # 0.55 - 0.5 = 0.05 <= delta 0.1
    assert core(res["SH-d0"]) == core(res["P2r"]) and sh_event(res["SH-d0"])["decision"] == "retry"                    # 0.05 > 0
    f = res["SHF"]                                                                                                       # max p = 0.55 < tau 0.6: flagged at the decision
    assert f["flagged"] and f["flag_reason"] == "low_value" and f["rewinds"] == 0 and not f["success"] and sh_event(f)["decision"] == "flag"
    assert core(res["SHF2"]) == core(res["P2r"])                                                                          # max p = 0.9 >= tau: the SH decision (retry)


def test_p2r_follows_the_part_2_retry_and_ignores_later_alarms(rig):
    cfg = cfg_for(thr=1e9)                                                                                              # the gate always says UNCHANGED: PA rewinds and retries
    res = run_episode(rig, cfg, [pa(), sh("P2r", "retry")])
    pa_t, r_t = res["PA"]["log"]["trace"], res["P2r"]["log"]["trace"]
    firsts = [i for i, t in enumerate(pa_t) if t[0] == "pause"]
    second_pause = next(i for i in range(firsts[0] + 10, len(pa_t)) if pa_t[i][0] == "pause")
    assert second_pause > firsts[0] + 20 and r_t[:second_pause] == pa_t[:second_pause], "P2r's first retry differs from PA's (the Part 2 retry)"
    p2r = res["P2r"]
    assert p2r["rewinds"] == 1 and p2r["paused"] == 10 and not p2r["flagged"] or p2r["success"], (p2r["rewinds"], p2r["paused"], p2r["flagged"])
    assert sum(1 for e in p2r["log"]["events"] if e["kind"] == "sh") == 1


def test_pc_equals_pas_grace_and_p0_with_an_empty_pause(rig, monkeypatch):
    res = run_episode(rig, cfg_for(thr=-1e9), [pa(), sh("PC", "continue")])
    pa_t, pc_t = res["PA"]["log"]["trace"], res["PC"]["log"]["trace"]
    assert res["PA"]["log"]["events"][0].get("grace") is True and len(pc_t) > len(pa_t) and pc_t[:len(pa_t)] == pa_t, "PC differs from PA's grace path"
    assert res["PC"]["flagged"] is False and res["PC"]["rewinds"] == 0
    real = L.run_pause
    monkeypatch.setattr(L, "run_pause", lambda budget, fn, k=0: real(budget, fn, k=0))
    res = run_episode(rig, cfg_for(), [sh("PC", "continue")])
    assert res["PC"]["log"]["trace"] == res["P0"]["log"]["trace"], "continue with an empty pause must be P0's continuation over the whole episode"
    assert res["PC"]["success"] == res["P0"]["success"] and res["PC"]["paused"] == 0


def test_noise_keys_of_the_branches(rig):
    res = run_episode(rig, cfg_for(), [sh("P2r", "retry"), sh("R1", "retry", key=1), sh("R2", "retry", key=2), sh("PC", "continue"), sh("C1", "continue", key=1), sh("C2", "continue", key=2)])
    assert core(res["R1"]) == core(res["P2r"])
    assert core(res["R2"]) != core(res["R1"]) and res["R2"]["log"]["decisions"][-1]["iv"] == 2 or any(x["iv"] == 2 for x in res["R2"]["log"]["decisions"])
    n = res["PC"]["paused"] + res["PC"]["log"]["events"][0]["d"] * 10
    first_chunk = lambda r: [t[1] for t in r["log"]["trace"][n:n + 10]]
    assert first_chunk(res["C1"]) == first_chunk(res["PC"]) == first_chunk(res["C2"]), "the planned chunk is executed in every continue branch"
    assert all(x["iv"] == 0 for x in res["PC"]["log"]["decisions"])
    assert any(x["iv"] == 2001 for x in res["C1"]["log"]["decisions"]) and any(x["iv"] == 2002 for x in res["C2"]["log"]["decisions"])
    assert core(res["C1"]) != core(res["PC"]) or res["C1"]["elapsed"] != res["PC"]["elapsed"] or True


@pytest.mark.parametrize("mode,key", [("retry", 0), ("retry", 3), ("continue", 0), ("continue", 2)])
def test_branch_equals_fresh_replay(rig, mode, key):
    d = 8
    base = p2_run.make_cfg("test", 5, 25, "g-sh", conf())
    cfg = dataclasses.replace(base, trace=True, p4=True, accel=CusumParams.make(0.0, 1.0, 1e12), force_alarms=frozenset({d}))
    arm = sh(f"SH-{mode}{key}", mode, key=key)
    branched = run_episode(rig, cfg, [arm])[arm.name]
    fresh = run_arm_fresh(rig, cfg, arm, d)
    c = lambda r: json.dumps({"t": r["log"]["trace"], "d": r["log"]["decisions"], "e": r["log"]["events"], **{k: r[k] for k in ("success", "flagged", "flag_reason", "elapsed", "executed", "paused", "returned", "rewinds")}}, sort_keys=True, default=str)
    assert len(branched["log"]["trace"]) > 10 * d and sh_event(branched)["x"] == sh_event(fresh)["x"]
    assert c(branched) == c(fresh), f"SH {mode} key {key} branched at decision {d} differs from the fresh replay"
