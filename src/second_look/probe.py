"""Part 3 probe (results/part3/notes/p3_probe_plan.txt): "look around at the point of disagreement". NO home pose and NO rewind: everything happens at the exact alarm pose X.

At the first alarm of an arm the loop pauses 10 steps and runs the DeltaDINO check as in Part 2 / PA (src/second_look/loop.py). CHANGED: the grace of PA (reset the CUSUM, continue; any later alarm flags at once).
UNCHANGED: the arm is at X (the pose where the disagreement fired, after the pause) and the arm kind decides:

  PS   (stay)        stay at X, reset the CUSUM, execute the already planned chunk; flag at the next alarm.
  PNr  (random)      candidates = the 16 small end-effector moves around X (below), collision-free ones only; move to the FIRST candidate of a seeded permutation (a uniformly random one), plan there, continue.
  PNd  (disagreement, main) same candidates, visit up to K = 3 in the seeded order; at each: SmolVLA plans WITHOUT executing, the WITHIN-POSE disagreement D_w of the 8 sampled chunks (below), STAC against the
       previous decision (reported only) and accel are computed; accept the first candidate with D_w <= 0.7 x D_w(X) (relative; amendment 1 of results/part3/notes/p3_probe_plan.txt); if none of the visited passes, go to the
       visited one with the lowest D_w. D_w at X itself is computed first ("before" value).
  PNa  (accel)       as PNd with accel <= accel mu0 + 0.5 sigma (frozen CAL values) and the lowest accel as the fallback.
  PNp  (policy)      candidates = the poses reached by executing the first 3 actions of each of the 8 samples planned AT X (in a snapshot branch; gripper command held; discarded when an object moves
                     >= 1 cm / 5 deg, the robot penetrates > 1 cm, or a joint limit is hit); visit up to 3, select by D_w as PNd.
In every probe arm any LATER alarm flags at once (no pause, no gate), like PA after its grace.

Candidates (16): the 8 moves +-2 cm in world x, y, z and +-10 deg of wrist yaw (rotation about the world z axis), each at full and at half size (interpretation: "16 small moves" = the 8 specified moves x 2 sizes).
A candidate is kept when a simulated trial move from X (snapshot branch, restored afterwards; the simulator stands in for a collision checker, NOT charged) ends within tolerance in <= MAX_MOVE_STEPS control
steps, with robot penetration <= 1 cm all along, no object moved (label_v2 between the state at X and after the trial: < 1 cm and < 5 deg), joints inside their limits (margin 0.02 rad), no task success.

Moves are closed-loop OSC servo moves with the gripper command held at its last value; they are real robot time. STAC / accel / plan keys: noise key intervention index IV_X (X), IV_HAND + c (hand candidate c),
IV_POL + j (policy candidate j); ks 0 = the executed chunk, 1..8 = the samples.
D_w (within-pose disagreement, amendment 1) = the mean over the 28 pairs of the 8 sampled chunks of the per-step Euclidean distance over the 6 pose dimensions, averaged over the executed window (first 10 steps),
in the policy's normalised action space (the gripper is held in every probe move, so its dimension is left out). STAC against the previous decision (the loop's definition: biased RBF-MMD^2 between the previous
decision's 8 samples, continuation, and the 8 samples planned here, head) is computed and reported at X and at every visited pose but NOT used: a move alone inflates it.

CLOCK (control steps, budget 300): elapsed = executed + paused + returned + probed, probed = (control steps of every probe move) + 10 x (number of visited candidates) for PNd / PNa / PNp
  [= 10 for the plan at X, + 10 for every visited candidate's plan that is NOT the one executed; the executed plan is the decision's own plan, as in every arm] and = (steps of the one move) for PNr
  (its plan at X and the STAC values are instrumentation, never charged, never used) and 0 for PS. Planning without executing is charged 10 steps = the time of the decision it replaces.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from second_look import detectors, snapshot
from second_look.seeding import decision_noise

PROBE_KINDS = ("PS", "PNr", "PNd", "PNa", "PNp")
PLAN_COST = 10
K_VISIT = 3
N_SAMPLES = 8
POL_STEPS = 3
TRANS = 0.02
YAW = float(np.radians(10.0))
SIZES = (1.0, 0.5)
MOVE_TOL_POS = 0.004
MOVE_TOL_ROT = 0.04
MAX_MOVE_STEPS = 30
JOINT_MARGIN = 0.02
MAX_PEN = 0.01
ACCEPT_RATIO = 0.7                 # amendment 1: accept a candidate whose within-pose disagreement is <= 0.7 x that at the alarm pose X
EXEC_WINDOW = 10
IV_X, IV_HAND, IV_POL = 1000, 1001, 1101
BASE = [("+x", (1, 0, 0), 0.0), ("-x", (-1, 0, 0), 0.0), ("+y", (0, 1, 0), 0.0), ("-y", (0, -1, 0), 0.0), ("+z", (0, 0, 1), 0.0), ("-z", (0, 0, -1), 0.0), ("+yaw", (0, 0, 0), 1.0), ("-yaw", (0, 0, 0), -1.0)]
HAND = [(f"{n}@{s}", np.array(v, float) * TRANS * s, yaw * YAW * s) for s in SIZES for n, v, yaw in BASE]            # 16 candidates, index = position in this list


# ------------------------------------------------------------------------------------------------ geometry
def _md(renv):
    return snapshot._raw(renv.sim)


_SITE_CACHE: dict = {}


def _eef_site(renv, m) -> int:
    key = id(m)
    if key not in _SITE_CACHE:
        name = getattr(renv.robots[0].controller, "eef_name", None)
        if name is not None:
            _SITE_CACHE[key] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        else:
            sid = renv.robots[0].eef_site_id
            _SITE_CACHE[key] = int(sid["right"] if isinstance(sid, dict) else sid)
    return _SITE_CACHE[key]


def ee_pose(renv) -> tuple[np.ndarray, np.ndarray]:
    """End-effector position (3,) and rotation matrix (3, 3) in the world frame, from the simulator's kinematics (no side effect on the dynamics: mj_kinematics only refreshes derived arrays)."""
    m, d = _md(renv)
    mujoco.mj_kinematics(m, d)
    sid = _eef_site(renv, m)
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _joints_ok(renv) -> bool:
    m, d = _md(renv)
    qa, _ = snapshot.robot_addresses(m)
    qa = set(qa.tolist())
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_HINGE and int(m.jnt_qposadr[j]) in qa and m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            q = d.qpos[int(m.jnt_qposadr[j])]
            if q < lo + JOINT_MARGIN or q > hi - JOINT_MARGIN:
                return False
    return True


def _scale(renv) -> np.ndarray:
    c = renv.robots[0].controller
    return np.broadcast_to(np.asarray(c.output_max, float) / np.asarray(c.input_max, float), (6,)).copy()


def _servo(renv, g: float, target, step_fn, max_steps: int = MAX_MOVE_STEPS) -> tuple[int, bool]:
    """Closed-loop OSC move to target = (pos, R): delta action = clipped pose error / controller scale. `step_fn(action)` makes ONE control step and returns True to stop (success, budget). Returns (steps, reached)."""
    scale = _scale(renv)
    tp, tR = target
    n = 0
    while n < max_steps:
        pos, R = ee_pose(renv)
        ep, er = tp - pos, Rotation.from_matrix(tR @ R.T).as_rotvec()
        if np.linalg.norm(ep) < MOVE_TOL_POS and np.linalg.norm(er) < MOVE_TOL_ROT:
            return n, True
        a = np.zeros(7)
        a[:3] = np.clip(ep / scale[:3], -1.0, 1.0)
        a[3:6] = np.clip(er / scale[3:6], -1.0, 1.0)
        a[6] = g
        n += 1
        if step_fn(a):
            break
    pos, R = ee_pose(renv)
    return n, bool(np.linalg.norm(tp - pos) < MOVE_TOL_POS and np.linalg.norm(Rotation.from_matrix(tR @ R.T).as_rotvec()) < MOVE_TOL_ROT)


# ------------------------------------------------------------------------------------------------ simulated trials (screening; not charged, restored afterwards)
class _Trial:
    """Runs raw env steps in a snapshot branch and tracks penetration / joint limits / success; `finish` restores the snapshot."""

    def __init__(self, rig, st):
        from second_look import loop as L
        self.L, self.rig, self.renv = L, rig, rig.renv
        L.activate_env(self.renv)
        self.snap = snapshot.capture(rig.env)
        self.flat0 = L.get_flat(self.renv)
        self.m, self.d = _md(self.renv)
        self.reset()

    def reset(self):
        self.maxpen, self.joints_ok, self.success, self.steps = 0.0, True, False, 0

    def step(self, a) -> bool:
        L = self.L
        L.activate_env(self.renv)
        _, _, _, _, info = self.rig.env.step(np.asarray(a, np.float64))
        self.steps += 1
        self.maxpen = max(self.maxpen, snapshot.robot_penetration(self.m, self.d))
        self.joints_ok &= _joints_ok(self.renv)
        if info.get("is_success", False):
            self.success = True
            return True
        return False

    def verdict(self, reached: bool = True) -> dict:
        L = self.L
        flat1 = L.get_flat(self.renv)
        moved = int(self.rig.frames.oracle(self.flat0, flat1)["label"]) == 1
        why = [w for w, bad in (("not_reached", not reached), ("penetration", self.maxpen > MAX_PEN), ("joint_limit", not self.joints_ok), ("object_moved", moved), ("success", self.success)) if bad]
        pos, R = ee_pose(self.renv)
        return {"valid": not why, "why": why, "steps": self.steps, "max_penetration": float(self.maxpen), "pos": pos, "R": R}

    def restore(self):
        self.L.activate_env(self.renv)
        snapshot.restore(self.rig.env, self.snap)


def screen_hand(rig, st) -> list[dict]:
    """The 16 hand-made candidates around the current pose; each is trial-moved from X and the snapshot is restored after every trial."""
    tr = _Trial(rig, st)
    pos0, R0 = ee_pose(rig.renv)
    out = []
    try:
        for ci, (name, dp, dyaw) in enumerate(HAND):
            target = (pos0 + dp, _rz(dyaw) @ R0)
            tr.reset()
            steps, reached = _servo(rig.renv, st.g_last, target, tr.step)
            v = tr.verdict(reached)
            out.append({"idx": ci, "name": name, "target": target, "iv": IV_HAND + ci, **{k: v[k] for k in ("valid", "why", "steps", "max_penetration")}})
            tr.restore()
    finally:
        tr.restore()
    return out


def screen_pol(rig, cfg, st, samples: np.ndarray) -> list[dict]:
    """PNp: for each of the 8 samples planned at X execute its first POL_STEPS actions (gripper held) in a snapshot branch; the pose reached is a candidate."""
    import torch
    tr = _Trial(rig, st)
    out = []
    try:
        for j in range(samples.shape[0]):
            acts = []
            for t in range(POL_STEPS):
                a = rig.vla._post(torch.from_numpy(np.ascontiguousarray(samples[j, t], np.float32)).to(rig.vla.device)[None])
                a = np.asarray(a, np.float64).copy()
                a[6] = st.g_last
                acts.append(a)
            tr.reset()
            for a in acts:
                if tr.step(a):
                    break
            v = tr.verdict(True)
            out.append({"idx": j, "name": f"pol{j}", "target": (v["pos"], v["R"]), "iv": IV_POL + j, **{k: v[k] for k in ("valid", "why", "steps", "max_penetration")}})
            tr.restore()
    finally:
        tr.restore()
    return out


# ------------------------------------------------------------------------------------------------ planning at a pose
def within_spread(samples: np.ndarray) -> float:
    """D_w: mean over the K(K-1)/2 pairs of the sampled chunks of the per-step L2 distance over the 6 pose dims, averaged over the executed window (normalised action space)."""
    x = np.asarray(samples, np.float64)[:, :EXEC_WINDOW, :6]
    iu = np.triu_indices(x.shape[0], 1)
    return float(np.linalg.norm(x[iu[0]] - x[iu[1]], axis=-1).mean())


def plan_at(rig, cfg, st, iv: int, d: int, prev_samples, *, samples: bool = True) -> dict:
    """Plan WITHOUT executing at the current observation: the executed chunk (key ks 0), accel of its denoise path, K = 8 resamples (ks 1..8) and STAC against the previous decision's samples."""
    from second_look import loop as L
    L.activate_env(rig.renv)
    noise = decision_noise(cfg.split, cfg.task, cfg.init, cfg.seed_id, iv, d, ks=(0,))
    plan, batch = rig.vla.plan(st.obs, rig.env.task_description, noise)
    out = {"actions": plan.actions, "accel": detectors.accel(plan.iterates, cfg.accel_scale), "stac": None, "dw": None, "samples": None}
    if samples:
        noises = decision_noise(cfg.split, cfg.task, cfg.init, cfg.seed_id, iv, d, ks=range(1, N_SAMPLES + 1))
        L.activate_env(rig.renv)
        smp = rig.vla.resample(batch, noises)
        out["samples"], out["stac"], out["dw"] = smp, detectors.stac(smp, prev_samples), within_spread(smp)
    return out


def ordered(cfg, d: int, label: str, idxs: list[int]) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(f"probe-order|{label}|{cfg.split}|{cfg.task}|{cfg.init}|{cfg.seed_id}|{d}".encode()).digest()[:8], "little")
    return [idxs[i] for i in np.random.default_rng(seed).permutation(len(idxs))]


def _grab(st, label: str) -> None:
    """Camera frames of the policy env at this pose (agentview + wrist), for the showcase of the probe."""
    px = (st.obs or {}).get("pixels", {})
    st.log.setdefault("probe_frames", {})[label] = {k: np.asarray(v).copy() for k, v in px.items()}


def _clean(visits: list[dict]) -> list[dict]:
    return [{k: v for k, v in vv.items() if k != "res"} for vv in visits]


def _f(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------------------------------------ a (charged) move
def do_move(rig, cfg, st, target) -> dict:
    """Move to `target` in real robot time: every control step is executed through the loop's env_step (trace, success check) and charged to the budget."""
    from second_look import loop as L
    flat0 = L.get_flat(rig.renv)
    state = {"status": "ok", "pen": 0.0}
    m, d = _md(rig.renv)

    def step(a) -> bool:
        if st.budget.exhausted:
            state["status"] = "budget"
            return True
        st.budget.charge_probe_step()
        if L.env_step(rig, cfg, st, a, "probe"):
            state["status"] = "success"
            return True
        state["pen"] = max(state["pen"], snapshot.robot_penetration(m, d))
        return False
    steps, reached = _servo(rig.renv, st.g_last, target, step)
    steps -= 1 if state["status"] == "budget" else 0                                 # the step that was refused was counted by the servo loop
    moved = int(rig.frames.oracle(flat0, L.get_flat(rig.renv))["label"]) == 1
    return {"status": state["status"], "steps": steps, "reached": reached, "object_moved": moved, "max_penetration": float(state["pen"])}


# ------------------------------------------------------------------------------------------------ the arm
def handle_unchanged(rig, cfg, st, arm, dec, memo):
    """Called by second_look.loop.handle_alarm after an UNCHANGED verdict for a probe kind. Returns ("continue", dec') or ("done", dec). The arm is at X (after the pause), st.obs is the observation there."""
    from second_look import loop as L
    assert cfg.stac is not None, "the probe needs the STAC parameters (configs/part2_frozen.yaml)"
    d = st.d
    entry = st.ring[-1]
    assert entry.d == d
    prev = entry.stac_prev
    obs_X, g, p0 = copy.deepcopy(st.obs), st.g_last, st.budget.probed
    thr_a = cfg.accel.mu0 + 0.5 * cfg.accel.sigma
    use_acc = arm.kind == "PNa"
    ev = {"kind": "probe", "arm_kind": arm.kind, "d": d, "elapsed_before": st.budget.elapsed, "thr_accel": thr_a, "accept_ratio": ACCEPT_RATIO, "score": "accel" if use_acc else "dw"}
    st.alarm_done = True
    st.cus_a.restore(0.0)
    if st.cus_s is not None:
        st.cus_s.restore(0.0)

    def cached(key, fn):
        if memo is None:
            return fn()
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    def finish(what, dec_out):
        ev.update(elapsed_after=st.budget.elapsed, probed=st.budget.probed - p0)
        st.log["events"].append(ev)
        return what, dec_out

    # the "before" value: the plan at X after the pause (instrumentation for PS and PNr; charged where the arm uses it)
    xr = cached(("probe_x", d), lambda: plan_at(rig, cfg, st, IV_X, d, prev))
    ev.update(x_dw=_f(xr["dw"]), x_stac=_f(xr["stac"]), x_accel=_f(xr["accel"]))
    thr = thr_a if use_acc else ACCEPT_RATIO * xr["dw"]                         # PN-dis / PN-pol: relative to the disagreement at X (no new calibration); PN-acc: absolute (frozen CAL)
    ev["thr_score"] = _f(thr)
    _grab(st, "X")
    if arm.kind == "PS":
        ev.update(selected=None, passed=None, post_dw=_f(xr["dw"]), post_stac=_f(xr["stac"]), post_accel=_f(xr["accel"]), n_valid=0, n_cand=0, visits=[])
        return finish("continue", dec)
    if arm.kind != "PNr":
        st.budget.charge_probe_plan(PLAN_COST)                                   # the plan at X is a plan that is not executed
        if st.budget.exhausted:
            st.cut_short = True
            L.end(st)
            return finish("done", dec)
    # candidates
    if arm.kind == "PNp":
        cands = screen_pol(rig, cfg, st, xr["samples"])
        label = "pol"
    else:
        cands = cached(("probe_hand",), lambda: screen_hand(rig, st))
        label = "hand"
    st.obs, st.g_last = copy.deepcopy(obs_X), g                                  # the trials restored the simulator; the observation at X is the stored one
    valid = {c["idx"]: c for c in cands if c["valid"]}
    ev.update(n_cand=len(cands), n_valid=len(valid), invalid={c["name"]: c["why"] for c in cands if not c["valid"]})
    if not valid:
        ev.update(selected=None, passed=False, post_dw=_f(xr["dw"]), post_stac=_f(xr["stac"]), post_accel=_f(xr["accel"]), visits=[], empty=True)
        return finish("continue", dec)                                           # nothing to move to: as PS (the plan charge, if any, stays)
    order = ordered(cfg, d, label, sorted(valid))
    ev["order"] = order[:K_VISIT]
    visits = []
    k_max = 1 if arm.kind == "PNr" else K_VISIT
    sel = None
    for ci in order[:k_max]:
        c = valid[ci]
        mv = do_move(rig, cfg, st, c["target"])
        if mv["status"] == "success":
            L.end(st, success=True)
            ev["visits"] = _clean(visits) + [{"idx": ci, **{k: mv[k] for k in ("steps", "object_moved")}, "success_during_move": True}]
            return finish("done", dec)
        if mv["status"] == "budget":
            st.cut_short = True
            L.end(st)
            ev["visits"] = _clean(visits) + [{"idx": ci, "steps": mv["steps"], "budget_during_move": True}]
            return finish("done", dec)
        res = plan_at(rig, cfg, st, c["iv"], d, prev)
        _grab(st, f"c{ci}")
        score = res["accel"] if use_acc else res["dw"]
        visits.append({"idx": ci, "name": c["name"], "move_steps": mv["steps"], "reached": mv["reached"], "object_moved": mv["object_moved"], "max_penetration": mv["max_penetration"],
                       "dw": _f(res["dw"]), "stac": _f(res["stac"]), "accel": _f(res["accel"]), "score": _f(score), "ratio_to_X": _f(res["dw"] / xr["dw"]), "pass": bool(score <= thr), "res": res})
        if arm.kind == "PNr":
            sel = 0
            break
        st.budget.charge_probe_plan(PLAN_COST)                                   # every visited plan is charged; the executed one is refunded below
        if score <= thr:
            sel = len(visits) - 1
            break
        if st.budget.exhausted:
            st.cut_short = True
            L.end(st)
            ev["visits"] = _clean(visits)
            return finish("done", dec)
    passed = sel is not None and visits[sel]["pass"]
    if sel is None:
        sel = int(np.argmin([v["score"] for v in visits]))
    chosen = visits[sel]
    if arm.kind != "PNr":
        st.budget.probed -= PLAN_COST                                            # the executed plan is the decision's own plan: not charged
    plan_exec = chosen["res"]
    if sel != len(visits) - 1:                                                   # go back to the chosen candidate; plan afresh at the arrival observation (the decision's own plan)
        mv = do_move(rig, cfg, st, valid[chosen["idx"]]["target"])
        if mv["status"] == "success":
            L.end(st, success=True)
            ev["visits"] = _clean(visits)
            return finish("done", dec)
        if mv["status"] == "budget":
            st.cut_short = True
            L.end(st)
            ev["visits"] = _clean(visits)
            return finish("done", dec)
        ev["final_move_steps"] = mv["steps"]
        plan_exec = plan_at(rig, cfg, st, valid[chosen["idx"]]["iv"], d, prev, samples=False)
    ev.update(visits=_clean(visits), selected=chosen["idx"], passed=bool(passed), post_dw=chosen["dw"], post_stac=chosen["stac"], post_accel=chosen["accel"],
              move_steps_total=int(sum(v["move_steps"] for v in visits) + ev.get("final_move_steps", 0)), n_visited=len(visits))
    if st.budget.exhausted:
        st.cut_short = True
        L.end(st)
        return finish("done", dec)
    return finish("continue", replace(dec, actions=plan_exec["actions"]))
