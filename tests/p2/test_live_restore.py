"""G3 (arm-only restore) and G5 (budget with real steps). Live: needs a Scene (~1.5 GB RAM). P2_LIVE=1 python -m pytest tests/p2/test_live_restore.py"""
import mujoco
import numpy as np
import pytest

from second_look import settle
from second_look.budget import Budget, run_pause
from second_look.snapshot import arm_only_restore, capture, robot_addresses, unpack_qpos_qvel
from livehelp import object_xyz, scene_for

pytestmark = pytest.mark.live
TASK = "open_the_middle_drawer_of_the_cabinet"
DEC = 10                                                                       # steps per decision


def _run_to(sc, S, A, n_decisions):
    """Replay the demo's own actions from step 0, snapshotting at the start of every decision. Nothing is touched in the first 5 decisions."""
    settle.restore(sc, S, A, 0)
    snaps = []
    for d in range(n_decisions):
        snaps.append(capture(sc.env, meta={"decision": d}))
        for a in A[d * DEC:(d + 1) * DEC]:
            settle.step(sc, a)
    return snaps


def test_g3_arm_only_restore_on_a_static_scene():
    with scene_for(TASK) as (sc, S, A):
        snaps = _run_to(sc, S, A, 5)
        qa, da = robot_addresses(sc.m)
        assert len(qa) >= 9                                                    # 7 arm joints + 2 finger joints
        for L in (1, 2, 3, 4):
            target = snaps[5 - L]
            objs_before = object_xyz(sc)
            out = arm_only_restore(sc.env, target, scene=sc)
            qp, qv = unpack_qpos_qvel(sc.m, target["mj_state"])
            assert np.array_equal(sc.d.qpos[qa], qp[qa]) and np.array_equal(sc.d.qvel[da], qv[da]), f"L={L}: robot joints not bitwise equal"
            ctl = sc.renv.robots[0].controller
            for name, v in target["controllers"]["r0.arm"].items():
                if isinstance(v, np.ndarray):
                    assert np.array_equal(getattr(ctl, name), v), f"L={L}: controller {name}"
            assert np.array_equal(sc.renv.robots[0].gripper.current_action, target["controllers"]["r0.gripper"]["current_action"])
            assert out["valid"] and out["penetration"] <= 0.01
            assert all(np.array_equal(objs_before[k], v) for k, v in object_xyz(sc).items()), "objects must not be touched by the restore"
            settle.pause(sc, 10, float(A[(5 - L) * DEC - 1][6]))               # hold for one decision
            moved = max(float(np.linalg.norm(v - objs_before[k])) for k, v in object_xyz(sc).items())
            assert moved < 1e-3, f"L={L}: an object moved {moved * 1000:.2f} mm in 10 steps after the restore"
            _run_to(sc, S, A, 5)                                               # back to decision 5 for the next L


def test_g3_restore_leaves_objects_and_clears_nothing_else():
    """Move a free object by hand, then restore: the object stays where it is, the arm goes back."""
    with scene_for(TASK) as (sc, S, A):
        snaps = _run_to(sc, S, A, 4)
        free = next(it for it in sc.registry if it.kind == "free")
        sc.d.qpos[free.qadr] += 0.25
        mujoco.mj_forward(sc.m, sc.d)
        x_before = sc.d.qpos[free.qadr:free.qadr + 7].copy()
        arm_only_restore(sc.env, snaps[1], scene=sc)
        assert np.array_equal(sc.d.qpos[free.qadr:free.qadr + 7], x_before)


def test_g3_penetration_makes_the_rewind_invalid():
    """Put a free object into the target arm pose: the restore must report a penetration > 1 cm and valid=False."""
    with scene_for(TASK) as (sc, S, A):
        snaps = _run_to(sc, S, A, 5)
        free = next(it for it in sc.registry if it.kind == "free")
        eef = np.array(snaps[1]["controllers"]["r0.arm"]["goal_pos"])
        sc.d.qpos[free.qadr:free.qadr + 3] = eef                               # object right at the target hand position
        out = arm_only_restore(sc.env, snaps[1], scene=sc)
        assert out["penetration"] > 0.0
        if out["penetration"] > 0.01:
            assert not out["valid"]


def test_g5_budget_with_two_rewinds_and_real_steps():
    with scene_for(TASK) as (sc, S, A):
        settle.restore(sc, S, A, 0)
        b = Budget()
        snaps, hist = {}, 0
        g = float(A[0][6])
        def step_exec(a):
            settle.step(sc, a); b.charge_step()
        for d in range(8):
            snaps[d] = capture(sc.env)
            for a in A[d * DEC:(d + 1) * DEC]:
                step_exec(a)
            if d in (3, 6):                                                    # two alarms: pause, rewind 2 decisions, return cost 20
                n, ok = run_pause(b, lambda: (settle.step(sc, np.concatenate([settle.PAUSE_ARM, [g]])), False)[1])
                assert n == 10 and not ok
                out = arm_only_restore(sc.env, snaps[(d + 1) - 2], scene=sc)
                b.charge_return(20)
                hist += 1
        assert b.executed == 80 and b.paused == 20 and b.returned == 40 and b.elapsed == 140 and b.rewinds == hist == 2


def test_g5_success_during_a_pause_is_counted():
    with scene_for(TASK) as (sc, S, A):
        settle.restore(sc, S, A, 0)
        b = Budget()
        count = []
        def step():
            settle.step(sc, np.concatenate([settle.PAUSE_ARM, [-1.0]]))
            count.append(1)
            return len(count) == 6                                              # a task predicate that becomes true on the 6th pause step
        n, ok = run_pause(b, step)
        assert (n, ok, b.paused) == (6, True, 6)
