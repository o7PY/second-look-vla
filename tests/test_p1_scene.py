"""P1 exit checks (SPEC §8):
 (a) restoring a state and reading labels twice gives identical values
 (b) rendering the same state twice is bitwise identical
 (c) a pure pause on a static scene labels unchanged
 (d) teleporting the bowl 3 cm labels changed
plus: label thresholds, geodesic angle, seed determinism, k-ablation consistency, robot mask coverage.
"""
import h5py
import numpy as np
import pytest

from second_look import render, settle
from second_look.sampler import Demo, Spec, build_pair, pair_record
from second_look.scene import (DEMO_DIR, Scene, compare, label_from_deltas, label_pair, quat_angle_deg, task_names)

TASK = "put_the_bowl_on_the_plate"


@pytest.fixture(scope="module")
def world():
    sc = Scene(task_names().index(TASK))
    h = h5py.File(DEMO_DIR / f"{TASK}_demo.hdf5", "r")
    demo = Demo(sc, h, 0)
    return sc, render.Masks(sc), demo


def test_a_restore_and_read_twice(world):
    sc, _, demo = world
    for t in (0, 30, 60):
        settle.restore(sc, demo.states, demo.actions, t)
        a = sc.read()
        settle.restore(sc, demo.states, demo.actions, t)
        b = sc.read()
        assert all(np.array_equal(a[k], b[k]) for k in a)
        assert not label_pair(sc, a, b)["label"] and not label_pair(sc, a, b)["gray"]


def test_b_render_bitwise_identical(world):
    sc, masks, demo = world
    settle.restore(sc, demo.states, demo.actions, 40)
    f1 = render.render_frame(sc, masks)
    settle.restore(sc, demo.states, demo.actions, 10)
    render.render_frame(sc, masks)                      # disturb the context in between
    settle.restore(sc, demo.states, demo.actions, 40)
    f2 = render.render_frame(sc, masks)
    for k in f1:
        assert np.array_equal(f1[k], f2[k]), k


def test_c_pause_on_static_scene_is_unchanged(world):
    sc, _, demo = world
    # t=0: nothing is moving yet (arm at rest, objects at rest)
    g = settle.restore(sc, demo.states, demo.actions, 0)
    r0 = sc.read()
    settle.pause(sc, 10, g)
    lab = label_pair(sc, r0, sc.read())
    assert lab["label"] == 0 and not lab["gray"], lab["max_trans"]


def test_d_teleport_bowl_3cm_is_changed(world):
    sc, _, demo = world
    settle.restore(sc, demo.states, demo.actions, 0)
    r0 = sc.read()
    bowl = next(it for it in sc.registry if "bowl" in it.name)
    sc.d.qpos[bowl.qadr] += 0.03
    import mujoco
    mujoco.mj_forward(sc.m, sc.d)
    lab = label_pair(sc, r0, sc.read())
    assert lab["label"] == 1 and bowl.name in lab["changed_items"]


def test_thresholds_and_gray(world):
    sc, _, demo = world
    settle.restore(sc, demo.states, demo.actions, 0)
    r0 = sc.read()
    bowl = next(it for it in sc.registry if "bowl" in it.name)

    def shifted(dx):
        r = {k: v.copy() for k, v in r0.items()}
        r[bowl.name][0] += dx
        return label_from_deltas(compare(sc.registry, r0, r))
    assert shifted(0.004) == (0, False)                  # < 0.5 cm: unchanged
    assert shifted(0.0075) == (0, True)                  # 0.5-1.0 cm: gray
    assert shifted(0.0125) == (1, False)                 # > 1.0 cm: changed


def test_geodesic_angle():
    q0 = np.array([1.0, 0, 0, 0])
    for deg in (1.0, 5.0, 90.0):
        h = np.radians(deg) / 2
        q = np.array([np.cos(h), 0, 0, np.sin(h)])
        assert quat_angle_deg(q0, q) == pytest.approx(deg, abs=1e-6)
        assert quat_angle_deg(q0, -q) == pytest.approx(deg, abs=1e-6)       # |<q0,q1>| handles double cover


def test_registry_excludes_robot(world):
    sc, _, _ = world
    names = [sc.body_names[it.body] for it in sc.registry]
    assert names and not any(n.startswith(("robot0_", "gripper0_", "mount0_")) for n in names)
    kinds = {it.kind for it in sc.registry}
    assert {"free", "slide", "hinge"} <= kinds


def test_robot_mask_nonempty_and_dilates(world):
    sc, masks, demo = world
    settle.restore(sc, demo.states, demo.actions, 40)
    f = render.render_frame(sc, masks)
    assert f["robot"].sum() > 500
    arm = render.arm_patches(f["robot"], f["robot"])
    assert arm.sum() > render.to_patch_mask(f["robot"]).sum()


def test_seed_is_deterministic_and_63bit():
    s = Spec("S2", 3, 7, 50, 10, 10, "G1", 2)
    assert s.seed == Spec("S2", 3, 7, 50, 10, 10, "G1", 2).seed
    assert s.seed != Spec("S2", 3, 7, 50, 10, 10, "G1", 3).seed
    assert 0 <= s.seed < 2**63


def test_build_pair_deterministic_and_k_consistent(world):
    sc, masks, demo = world
    spec = Spec("S2", 8, 0, 45, 10, 10, "G1", 1)
    a = build_pair(sc, masks, demo, spec)
    b = build_pair(sc, masks, demo, spec)
    assert a and b
    for key in ("ref", "set", "arm_ref", "arm_set", "chg_ref", "chg_set"):
        assert np.array_equal(pair_record(a, 10)[0][key], pair_record(b, 10)[0][key]), key
    # the k-ablation re-render at k in (0,5,10,20) must reproduce the main k=10 frame bitwise
    c = build_pair(sc, masks, demo, spec, ks=(0, 5, 10, 20))
    assert np.array_equal(c["frames"][10]["rgb"], a["frames"][10]["rgb"])
    assert np.array_equal(c["frames"][10]["robot"], a["frames"][10]["robot"])
