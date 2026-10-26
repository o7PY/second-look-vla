"""In-memory snapshots of a running LIBERO episode and the two ways back (SPEC_part2.md §2.3, §6 P2.2).

`capture` / `restore` / `policy_state` / `set_policy_state` are the old project's d2r/snapshot.py (src/second_look/_legacy_d2r/snapshot.py), same contents: MuJoCo
integration state, the model fields a reset may change, the Python-side controller state (OSC goal, gripper integrator), robosuite's counters and observable
caches, and the policy's action queue. `restore` puts EVERYTHING back (used by the branching of arms, which must equal a fresh replay bitwise, G2).

`arm_only_restore` is new and is what a rewind means in Part 2: the arm goes back, the world stays as it is now.
  copied from the snapshot:  qpos / qvel of every robot joint (arm joints and the two finger joints; a joint is robot if its body is robot0_/gripper0_/mount0_),
                             the numeric state of the arm controller and of the gripper model (OSC goal pose, gripper command integrator, ...)
  cleared:                   the policy's action queue
  NOT touched:               every free-object / drawer / knob state, the clock (the runner owns it), time, warm start, the counters of the env
  then:                      mj_forward, `scene.sync_visual()` if a scene is given (stove glow follows the knob that is there NOW), and a forced refresh of
                             the observables so the next policy observation shows the current scene with the arm at the target pose.
Validity (§2.3): if a robot geom penetrates a non-robot geom by more than 1 cm after the restore (the arm landed inside an object that moved) the rewind is
invalid; the caller flags the episode and counts it.
"""
from __future__ import annotations

import copy
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from second_look.scene import ROBOT_PREFIXES

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
MODEL_FIELDS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "geom_pos", "geom_quat",
                "geom_size", "geom_rgba", "cam_pos", "cam_quat", "cam_fovy", "light_pos", "light_dir",
                "light_diffuse", "light_ambient", "light_specular", "site_pos", "site_quat", "site_rgba", "mat_rgba")          # site_rgba: stove burner glow (the old d2r list missed it)
_SCALAR = (int, float, bool, np.integer, np.floating, np.bool_)
MAX_PENETRATION = 0.01          # metres: a rewind that leaves a robot geom deeper than this inside something else is invalid


def robosuite_env(env: Any) -> Any:
    """Unwrap a lerobot ``LiberoEnv`` / LIBERO ``OffScreenRenderEnv`` / `second_look.scene.Scene.env` to the robosuite env (the one owning timestep / robots)."""
    e = env
    while not (hasattr(e, "robots") and hasattr(e, "timestep")):
        nxt = getattr(e, "_env", None) or getattr(e, "env", None)
        if nxt is None:
            raise TypeError(f"could not find a robosuite env inside {type(env).__name__}")
        e = nxt
    return e


def _raw(sim: Any) -> tuple[Any, Any]:
    return sim.model._model, sim.data._data


def get_model_params(sim: Any) -> dict[str, np.ndarray]:
    m, _ = _raw(sim)
    return {f: np.array(getattr(m, f)) for f in MODEL_FIELDS if hasattr(m, f)}


def set_model_params(sim: Any, params: dict[str, np.ndarray]) -> None:
    m, _ = _raw(sim)
    for f, v in params.items():
        dst = getattr(m, f)
        if dst.shape != v.shape:
            raise ValueError(f"model field {f}: shape {dst.shape} != snapshot {v.shape} (different scene?)")
        dst[...] = v


def get_mj_state(sim: Any) -> np.ndarray:
    m, d = _raw(sim)
    out = np.empty(mujoco.mj_stateSize(m, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(m, d, out, STATE_SPEC)
    return out


def set_mj_state(sim: Any, state: np.ndarray) -> None:
    m, d = _raw(sim)
    mujoco.mj_setState(m, d, np.asarray(state, dtype=np.float64), STATE_SPEC)
    mujoco.mj_forward(m, d)


def _numeric_attrs(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(obj).items():
        if isinstance(v, np.ndarray) or isinstance(v, _SCALAR) or v is None:
            out[k] = copy.deepcopy(v)
        elif k.startswith("interpolator") and v is not None:
            out[k] = _numeric_attrs(v)
    return out


def _restore_attrs(obj: Any, saved: dict[str, Any]) -> None:
    for k, v in saved.items():
        if isinstance(v, dict):
            _restore_attrs(getattr(obj, k), v)
        else:
            setattr(obj, k, copy.deepcopy(v))


def _controllers(renv: Any) -> list[tuple[str, Any]]:
    out = []
    for i, robot in enumerate(renv.robots):
        out.append((f"r{i}.arm", robot.controller))
        if getattr(robot, "gripper", None) is not None:
            out.append((f"r{i}.gripper", robot.gripper))
    return out


def _vis_objects(renv: Any) -> dict[str, Any]:
    """Objects with a state-dependent visual site (the stove: burner glow, toggled by a Python flag in `object_properties["vis_site_names"]`)."""
    out = {}
    for name in dict.fromkeys(getattr(renv, "visualization_sites_list", [])):
        obj = renv.get_object(name)
        if hasattr(obj, "turn_on"):
            out[name] = obj
    return out


def get_visual_state(renv: Any) -> dict[str, Any]:
    return {n: copy.deepcopy(o.object_properties["vis_site_names"]) for n, o in _vis_objects(renv).items()}


def set_visual_state(renv: Any, st: dict[str, Any]) -> None:
    for n, o in _vis_objects(renv).items():
        o.object_properties["vis_site_names"] = copy.deepcopy(st[n])


def capture(env: Any, *, policy: Any = None, meta: dict | None = None) -> dict[str, Any]:
    """Snapshot `env` (and optionally a LeRobot `policy`'s queues) into a dict."""
    renv = robosuite_env(env)
    snap = {
        "mj_state": get_mj_state(renv.sim),
        "model": get_model_params(renv.sim),
        "controllers": {name: _numeric_attrs(c) for name, c in _controllers(renv)},
        "env_counters": {"timestep": int(renv.timestep), "cur_time": float(renv.cur_time), "done": bool(renv.done)},
        "observables": {name: _numeric_attrs(o) for name, o in renv._observables.items()},
        "obs_cache": copy.deepcopy(renv._obs_cache),
        "visual": get_visual_state(renv),
        "meta": dict(meta or {}),
    }
    if policy is not None:
        snap["policy"] = policy_state(policy)
    return snap


def restore(env: Any, snap: dict[str, Any], *, policy: Any = None) -> None:
    """Full restore (everything the snapshot holds). Used for branching, NOT for a rewind."""
    renv = robosuite_env(env)
    set_model_params(renv.sim, snap["model"])
    set_mj_state(renv.sim, snap["mj_state"])
    for name, c in _controllers(renv):
        _restore_attrs(c, snap["controllers"][name])
    for k, v in snap["env_counters"].items():
        setattr(renv, k, v)
    for name, o in renv._observables.items():
        _restore_attrs(o, snap["observables"][name])
    renv._obs_cache = copy.deepcopy(snap["obs_cache"])
    set_visual_state(renv, snap["visual"])
    if policy is not None and "policy" in snap:
        set_policy_state(policy, snap["policy"])


def observation(env: Any) -> dict:
    """The observation the policy saw at the snapshot (cached observable values), formatted by lerobot's LiberoEnv. Not force_update."""
    return env._format_raw_obs(robosuite_env(env)._get_observations())


def policy_state(policy: Any) -> dict[str, Any]:
    queues = getattr(policy, "_queues", None) or {}
    return {k: {"maxlen": dq.maxlen, "items": [t.detach().cpu().clone() for t in dq]} for k, dq in queues.items()}


def set_policy_state(policy: Any, st: dict[str, Any]) -> None:
    dev = next(policy.parameters()).device
    policy._queues = {k: deque([t.to(dev) for t in v["items"]], maxlen=v["maxlen"]) for k, v in st.items()}


def clear_policy_queue(policy: Any) -> None:
    """Empty every action queue (the next `select_action` plans a new chunk)."""
    for dq in (getattr(policy, "_queues", None) or {}).values():
        dq.clear()


def save(path: Path | str, snap: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rest = {k: v for k, v in snap.items() if k != "mj_state"}
    np.savez_compressed(path, mj_state=snap["mj_state"], extra=np.frombuffer(pickle.dumps(rest, protocol=5), dtype=np.uint8))
    return path


def load(path: Path | str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        snap = pickle.loads(z["extra"].tobytes())
        snap["mj_state"] = z["mj_state"].copy()
    return snap


# ------------------------------------------------------------------------------------------------ arm-only restore (new in Part 2)
def _is_robot_body(m: mujoco.MjModel, b: int) -> bool:
    return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(ROBOT_PREFIXES)


def robot_addresses(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """(qpos addresses, dof addresses) of every robot joint (hinge / slide on a robot0_ / gripper0_ / mount0_ body)."""
    qa, da = [], []
    for j in range(m.njnt):
        if _is_robot_body(m, int(m.jnt_bodyid[j])) and int(m.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qa.append(int(m.jnt_qposadr[j]))
            da.append(int(m.jnt_dofadr[j]))
    return np.array(qa, dtype=np.int64), np.array(da, dtype=np.int64)


def unpack_qpos_qvel(m: mujoco.MjModel, mj_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel stored inside a packed `mj_state` (layout-independent: read through a scratch MjData)."""
    scratch = mujoco.MjData(m)
    mujoco.mj_setState(m, scratch, np.asarray(mj_state, dtype=np.float64), STATE_SPEC)
    return scratch.qpos.copy(), scratch.qvel.copy()


def robot_penetration(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """Deepest penetration (metres, >= 0) between a robot geom and a non-robot geom among the current contacts (call after mj_forward)."""
    n = int(d.ncon)
    if n == 0:
        return 0.0
    is_rob = np.array([_is_robot_body(m, int(b)) for b in m.geom_bodyid], dtype=bool)
    g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
    cross = is_rob[g1] != is_rob[g2]
    return float(max(0.0, -d.contact.dist[:n][cross].min())) if cross.any() else 0.0


def arm_only_restore(env: Any, snap: dict[str, Any], *, policy: Any = None, scene: Any = None) -> dict:
    """Put the ARM back to the snapshot; leave the world. Returns {"penetration": metres, "valid": bool}. See the module docstring for what is copied."""
    renv = robosuite_env(env)
    sim = renv.sim
    m, d = _raw(sim)
    qp, qv = unpack_qpos_qvel(m, snap["mj_state"])
    qa, da = robot_addresses(m)
    d.qpos[qa] = qp[qa]
    d.qvel[da] = qv[da]
    mujoco.mj_forward(m, d)
    for name, c in _controllers(renv):
        _restore_attrs(c, snap["controllers"][name])
    if scene is not None:
        scene.sync_visual()
    renv._get_observations(force_update=True)
    if policy is not None:
        clear_policy_queue(policy)
    pen = robot_penetration(m, d)
    return {"penetration": pen, "valid": pen <= MAX_PENETRATION}
