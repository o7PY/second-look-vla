"""Disturbances (G1-G5) and hard negatives (H1-H4), SPEC §4.3.

Each generator runs at demo step ``t`` right after the state is restored and BEFORE the k-step pause.
It may edit velocities / joint positions or drive the arm with a scripted motion, and returns a dict
``{"g": gripper command for the pause, "check": optional post-hoc requirement}`` or ``None`` to reject
(e.g. "drop" when nothing is held). The label is never taken from the generator's intent; it is
re-measured from the simulator by the caller.
"""

from __future__ import annotations

import mujoco
import numpy as np

from second_look import settle
from second_look.scene import Scene

GENERATORS = ("G1", "G2", "G3", "G4", "G5")
HARD_NEGATIVES = ("H1", "H2", "H3", "H4")
TALL = ("wine_bottle", "cream_cheese")


# ---- helpers ------------------------------------------------------------------------------------
def free_items(scene: Scene, exclude: set[str] = frozenset()):
    return [it for it in scene.registry if it.kind == "free" and it.name not in exclude]


def eef_pos(scene: Scene) -> np.ndarray:
    ctrl = scene.renv.robots[0].controller
    ctrl.update(force=True)
    return ctrl.ee_pos.copy()


def obj_pos(scene: Scene, it) -> np.ndarray:
    return scene.d.qpos[it.qadr:it.qadr + 3].copy()


def drive(scene: Scene, direction: np.ndarray, n: int, g: float, speed: float, aim=None) -> None:
    """n steps of arm motion along ``direction`` (re-aimed each step at ``aim`` if given)."""
    for _ in range(n):
        v = direction
        if aim is not None:
            v = aim() - eef_pos(scene)
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            break
        a = np.zeros(7)
        a[:3] = np.clip(v / nv * speed, -1, 1)
        a[6] = g
        settle.step(scene, a)


def move_to(scene: Scene, target: np.ndarray, g: float, max_steps: int, speed: float = 0.8, tol: float = 0.01) -> None:
    for _ in range(max_steps):
        err = target - eef_pos(scene)
        if np.linalg.norm(err) < tol:
            break
        a = np.zeros(7)
        a[:3] = np.clip(err / 0.05, -1, 1) * speed
        a[6] = g
        settle.step(scene, a)


def unit_xy(rng: np.random.Generator) -> np.ndarray:
    th = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(th), np.sin(th), 0.0])


def _dofadr(scene: Scene, it) -> int:
    return int(scene.m.jnt_dofadr[it.jnt])


# ---- positives ----------------------------------------------------------------------------------
def g1_push(scene, demo, rng, t, g):
    cands = free_items(scene, set(settle.holding(scene)))
    if not cands:
        return None
    it = cands[rng.integers(len(cands))]
    v = unit_xy(rng) * rng.uniform(0.1, 0.5)
    a = _dofadr(scene, it)
    scene.d.qvel[a:a + 3] = v
    return {"g": g}


def g2_topple(scene, demo, rng, t, g):
    cands = [it for it in free_items(scene, set(settle.holding(scene))) if any(s in it.name for s in TALL)]
    if not cands:
        return None
    it = cands[rng.integers(len(cands))]
    w_world = unit_xy(rng) * rng.uniform(3, 10)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, scene.d.qpos[it.qadr + 3:it.qadr + 7].copy())
    w_local = R.reshape(3, 3).T @ w_world          # free-joint angular qvel is in the body frame
    a = _dofadr(scene, it)
    scene.d.qvel[a + 3:a + 6] = w_local
    return {"g": g}


def g3_drop(scene, demo, rng, t, g):
    if not settle.holding(scene):
        return None
    scene.renv.robots[0].gripper.current_action = np.array([1.0, -1.0])   # fully open
    return {"g": -1.0}


def g4_knock(scene, demo, rng, t, g):
    cands = free_items(scene)
    it = cands[rng.integers(len(cands))]
    drive(scene, None, int(rng.integers(10, 31)), g, float(rng.uniform(0.4, 1.0)), aim=lambda: obj_pos(scene, it))
    return {"g": g}


def g5_nudge(scene, demo, rng, t, g):
    arts = [it for it in scene.registry if it.kind in ("slide", "hinge")]
    if not arts:
        return None
    it = arts[rng.integers(len(arts))]
    lo, hi = scene.m.jnt_range[it.jnt]
    mag = rng.uniform(0.01, 0.05) if it.kind == "slide" else np.radians(rng.uniform(10, 45))
    q = scene.d.qpos[it.qadr]
    sign = rng.choice([-1.0, 1.0])
    if not lo <= q + sign * mag <= hi:
        sign = -sign
    scene.d.qpos[it.qadr] = float(np.clip(q + sign * mag, lo, hi))
    scene.d.qvel[_dofadr(scene, it)] = 0.0
    mujoco.mj_forward(scene.m, scene.d)
    return {"g": g}


# ---- hard negatives -----------------------------------------------------------------------------
def h1_arm_only(scene, demo, rng, t, g):
    if settle.holding(scene):
        return None                                   # carrying an object would move it
    d = unit_xy(rng) * rng.uniform(0.2, 1.0)
    d[2] = rng.uniform(0.2, 1.0)                      # biased upward: away from the table
    drive(scene, d / np.linalg.norm(d), int(rng.integers(10, 31)), g, float(rng.uniform(0.5, 1.0)))
    return {"g": g}


def h2_occluder(scene, demo, rng, t, g):
    if settle.holding(scene):
        return None
    cands = free_items(scene)
    it = cands[rng.integers(len(cands))]
    cam = scene.sim.model.camera_name2id("agentview")
    cam_pos = scene.d.cam_xpos[cam].copy()
    o = obj_pos(scene, it)
    ray = cam_pos - o
    ray /= np.linalg.norm(ray)                        # object -> camera
    target = o + ray * rng.uniform(0.14, 0.24)        # hand on the line of sight
    target[2] = max(target[2], o[2] + 0.12)
    move_to(scene, target, g, max_steps=40)
    return {"g": g, "check": ("occlusion", it.name)}


def h3_jitter(scene, demo, rng, t, g):
    cands = free_items(scene, set(settle.holding(scene)))
    if not cands:
        return None
    it = cands[rng.integers(len(cands))]
    q = scene.d.qpos
    q[it.qadr:it.qadr + 3] += unit_xy(rng) * rng.uniform(0.001, 0.004)
    ang = np.radians(rng.uniform(0.5, 2.0)) * rng.choice([-1, 1])
    dq = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, q[it.qadr + 3:it.qadr + 7].copy())
    q[it.qadr + 3:it.qadr + 7] = out / np.linalg.norm(out)
    a = _dofadr(scene, it)
    scene.d.qvel[a:a + 6] = 0.0
    mujoco.mj_forward(scene.m, scene.d)
    return {"g": g}


def h4_static(scene, demo, rng, t, g):
    return {"g": g}


def _far_direction(rng, back_bias: bool = True) -> np.ndarray:
    """Horizontal unit vector aimed at the places training never saw changes (R2-B): mostly the back wall (-x, behind the table, towards the robot),
    the table sides (+-y) and the front edge (+x), with +-15 deg jitter. World axes (checked): the agentview camera is at +x looking towards -x; the robot base is at x = -0.66."""
    base = [(-1.0, 0.0), (-1.0, 1.0), (-1.0, -1.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0)] if back_bias else [(-1.0, 0.0)]
    w = np.array([0.35, 0.1, 0.1, 0.15, 0.15, 0.15]) if back_bias else np.array([1.0])
    dx, dy = base[int(rng.choice(len(base), p=w / w.sum()))]
    th = np.arctan2(dy, dx) + np.radians(rng.uniform(-15, 15))
    return np.array([np.cos(th), np.sin(th), 0.0])


FAR_SPEED = (0.6, 1.2)          # m/s; the spec range. At this speed objects slide a median 4 cm (friction) and never reach the static region (measured), so the probe builder raises it


def g1_push_far(scene, demo, rng, t, g):
    """G1-far: a free object pushed at U(0.6, 1.2) m/s towards the back wall / table edges (R2-B probe; never used for training)."""
    cands = free_items(scene, set(settle.holding(scene)))
    if not cands:
        return None
    it = cands[rng.integers(len(cands))]
    a = _dofadr(scene, it)
    scene.d.qvel[a:a + 3] = _far_direction(rng) * rng.uniform(*FAR_SPEED)
    return {"g": g}


def g2_topple_far(scene, demo, rng, t, g):
    """G2-far: a tall object toppled towards the back (-x): angular velocity about the horizontal axis that tips +z towards the chosen direction."""
    cands = [it for it in free_items(scene, set(settle.holding(scene))) if any(s_ in it.name for s_ in TALL)]
    if not cands:
        return None
    it = cands[rng.integers(len(cands))]
    d = _far_direction(rng, back_bias=True)
    axis = np.cross(np.array([0.0, 0.0, 1.0]), d)                       # rotating z towards d: rotation about z x d
    w_world = axis / np.linalg.norm(axis) * rng.uniform(3, 10)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, scene.d.qpos[it.qadr + 3:it.qadr + 7].copy())
    scene.d.qvel[_dofadr(scene, it) + 3:_dofadr(scene, it) + 6] = R.reshape(3, 3).T @ w_world
    return {"g": g}


REGISTRY = {"G1": g1_push, "G2": g2_topple, "G3": g3_drop, "G4": g4_knock, "G5": g5_nudge,
            "H1": h1_arm_only, "H2": h2_occluder, "H3": h3_jitter, "H4": h4_static,
            "S1": h4_static, "G1F": g1_push_far, "G2F": g2_topple_far}      # S1: plain human-demo pair (no disturbance); pause only
