"""R3: a render is a function of the restored physics state (burner glow and any other render state).
'Live render' = the render of the live environment after sim.forward(): right after mj_step MuJoCo's derived kinematics lag the integrated
qpos by <= 0.06 mm (390 pixels differ on the stove task); forward() removes that, so the test isolates the render-state question."""
import contextlib
import gc

import h5py
import numpy as np
import pytest

from second_look import render, settle
from second_look.scene import DEMO_DIR, Scene, task_names

NAMES = task_names()
STOVE = "turn_on_the_stove"


def _load(scene, task_name):
    with h5py.File(DEMO_DIR / f"{task_name}_demo.hdf5", "r") as h:
        g = h["data/demo_0"]
        scene.load_demo_model(g.attrs["model_file"])
        return g.attrs["model_file"], g["states"][:], g["actions"][:]


@contextlib.contextmanager
def scene_for(task_name):
    """One Scene at a time, always closed: a Scene is ~1.5 GB for the first and 0.1-0.4 GB for each further one (measured), and an earlier version of this
    test kept ten alive and, with the machine already short of RAM, took the desktop down (global OOM, 2026-10-06 14:23)."""
    sc = Scene(NAMES.index(task_name))
    try:
        yield sc, *_load(sc, task_name)
    finally:
        sc.env.close()
        del sc
        gc.collect()


def _live_render(scene):
    scene.sim.forward()
    return render.render_rgb(scene)


def _alpha(scene):
    return float(scene.sim.model.site_rgba[scene.sim.model.site_name2id("flat_stove_1_burner")][3])


@pytest.fixture(scope="module")
def stove():
    with scene_for(STOVE) as (sc, xml, S, A):
        knob = next(it for it in sc.registry if it.name == "flat_stove_1_button")
        yield sc, xml, S, A, knob


def test_burner_after_restore(stove):
    sc, xml, S, A, knob = stove
    settle.restore(sc, S, A, 0)
    off_live = _live_render(sc)
    for t in range(len(A)):                                           # live episode until the knob passes the 'on' angle
        settle.step(sc, A[t])
        if sc.d.qpos[knob.qadr] >= 0.55:
            break
    assert sc.d.qpos[knob.qadr] >= 0.5 and _alpha(sc) > 0, "the burner should glow live"
    on_state, on_live = sc.get_flat(), _live_render(sc)

    # (1) restore the 'on' state into a FRESH scene (closed again straight away)
    with scene_for(STOVE) as (fresh, _x, _S, _A):
        fresh.set_flat(on_state)
        assert np.array_equal(render.render_rgb(fresh), on_live)
    # (2) restore the 'on' state into a scene where the burner is off
    settle.restore(sc, S, A, 0); assert _alpha(sc) < 0
    sc.set_flat(on_state)
    assert np.array_equal(render.render_rgb(sc), on_live)
    # (3) restore the 'off' state into a scene where the burner is on (this is the stale-glow bug)
    assert _alpha(sc) > 0
    sc.set_flat(S[0])
    assert _alpha(sc) < 0 and np.array_equal(render.render_rgb(sc), off_live)
    # (4) the restored scene also steps consistently: one more step keeps the glow state a function of the knob
    sc.set_flat(on_state); settle.pause(sc, 1, -1.0)
    assert _alpha(sc) > 0
    settle.restore(sc, S, A, 0); settle.pause(sc, 1, -1.0)
    assert _alpha(sc) < 0


def test_burner_not_reset_by_plain_restore_would_fail_without_sync(stove):
    """Guards the guard: switching the sync off reproduces the stale glow, so the test above really tests the fix."""
    sc, xml, S, A, knob = stove
    for t in range(len(A)):
        settle.step(sc, A[t])
        if sc.d.qpos[knob.qadr] >= 0.55:
            break
    sc.sim.set_state_from_flattened(S[0]); sc.sim.forward()           # a restore WITHOUT sync_visual
    assert _alpha(sc) > 0 and sc.d.qpos[knob.qadr] < 0.1               # stale glow with the knob off
    sc.set_flat(S[0]); assert _alpha(sc) < 0                           # the fix removes it


def test_render_state_roundtrip():
    """20 random states across all tasks: restore -> render equals the live render bitwise. One scene alive at a time."""
    rng = np.random.default_rng(0)
    tasks = rng.integers(len(NAMES), size=20)
    for ti in sorted(set(int(t) for t in tasks)):
        with scene_for(NAMES[ti]) as (sc, xml, S, A):
            for _ in range(int((tasks == ti).sum())):
                t = int(rng.integers(5, len(A) - 5))
                settle.restore(sc, S, A, int(rng.integers(0, len(A) - 1)))     # leave the scene in some other state
                for k in range(int(rng.integers(1, 12))):                       # play on live for a few steps
                    settle.step(sc, A[min(t + k, len(A) - 1)])
                state, live = sc.get_flat(), _live_render(sc)
                settle.restore(sc, S, A, int(rng.integers(0, len(A) - 1)))     # ... elsewhere ...
                sc.set_flat(state)                                              # ... and back
                assert np.array_equal(render.render_rgb(sc), live), (NAMES[ti], t)


def test_fix_does_not_touch_non_stove_renders():
    with scene_for("put_the_bowl_on_the_plate") as (sc, xml, S, A):
        settle.restore(sc, S, A, 30); before = render.render_rgb(sc)
        sc.sync_visual(); sc.sync_visual()                               # idempotent
        assert np.array_equal(render.render_rgb(sc), before)
