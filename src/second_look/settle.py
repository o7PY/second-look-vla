"""Restore a demo moment and "pause the robot for k steps" (SPEC §3.2).

Hidden state that a plain qpos/qvel restore misses (robosuite 1.4):
  * the gripper model integrates ``current_action += [-1, 1] * 0.01 * sign(a_gripper)`` per control
    step, so we rebuild it from the demo's action history up to ``t``;
  * the OSC arm controller re-reads the live end-effector pose in ``set_goal`` each step (delta mode),
    so it has no state to restore beyond a forced ``update``.
"""

from __future__ import annotations

import numpy as np

from second_look.scene import Scene

GRIPPER_SPEED = 0.01
# LIBERO demos start with the gripper integrator saturated open. Measured: init [1,-1] reproduces the
# demo's finger trajectory to 3e-5 m; robosuite's own reset value (0) is wrong for these demos.
GRIPPER_INIT = np.array([1.0, -1.0])
PAUSE_ARM = np.zeros(6)


def gripper_integral(grip_cmds: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    """Replay robosuite's PandaGripper.format_action over a gripper-command history."""
    c = GRIPPER_INIT.copy() if init is None else init.copy()
    for g in grip_cmds:
        c = np.clip(c + np.array([-1.0, 1.0]) * GRIPPER_SPEED * np.sign(g), -1.0, 1.0)
    return c


def _robot(scene: Scene):
    return scene.renv.robots[0]


def restore(scene: Scene, states: np.ndarray, actions: np.ndarray, t: int, grip: np.ndarray | None = None) -> float:
    """Put the sim at episode step ``t`` (state before action ``t``). Returns the last gripper command.

    ``grip`` (T,2) = the recorded gripper-integrator state per step (policy rollouts record it); without it
    (human demos) the integrator is rebuilt from the action history."""
    scene.set_flat(states[t])
    last_g = float(actions[t - 1][6]) if t > 0 else -1.0
    _robot(scene).gripper.current_action = np.array(grip[t], dtype=np.float64) if grip is not None else gripper_integral(actions[:t, 6])
    sync(scene)
    return last_g


def sync(scene: Scene) -> None:
    """Make the controller / env counters consistent with the (just-restored) sim state."""
    r = scene.renv
    r.timestep = 0           # LIBERO horizon is per-episode; we step far fewer than it, but stay clean
    r.done = False
    _robot(scene).controller.update(force=True)


def step(scene: Scene, action: np.ndarray) -> None:
    scene.renv.step(np.asarray(action, dtype=np.float64))
    scene.renv.done = False


def pause(scene: Scene, k: int, g: float) -> None:
    """``k`` steps of [0,0,0,0,0,0,g]: arm holds still, gripper keeps its last command, physics runs."""
    a = np.concatenate([PAUSE_ARM, [g]])
    for _ in range(k):
        step(scene, a)


def holding(scene: Scene) -> list[str]:
    """Names of free objects in contact with both gripper fingers (a cheap 'is it held' test)."""
    import mujoco
    m, d = scene.m, scene.d
    fingers = {i for i in range(m.nbody) if scene.body_names[i].startswith("gripper0_") and "finger" in scene.body_names[i]}
    if not fingers:
        fingers = {i for i in range(m.nbody) if scene.body_names[i].startswith("gripper0_")}
    touch: dict[int, set] = {}
    free = {it.body: it for it in scene.registry if it.kind == "free"}
    for c in d.contact[: d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        for a, b in ((b1, b2), (b2, b1)):
            if a in fingers and b in free:
                touch.setdefault(b, set()).add(a)
    return [free[b].name for b, f in touch.items() if len(f) >= 2]
