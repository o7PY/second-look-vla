"""The closed loop: one episode, all arms, one process (SPEC_part2.md §2, §3, §6 P2.2 "episode-major branching").

The baseline P0 is played from reset. At every decision (= one SmolVLA re-plan = 10 steps) the loop snapshots the simulator and the detectors into a ring of the
last 12 decisions and plans the chunk with the keyed noise of that decision. When the FIRST alarm of a trigger (accel CUSUM, STAC CUSUM, or P5's random draw) is
reached, every arm that listens to that trigger is played from that point to its end, each from a full restore of the same snapshot, then the baseline is restored
and continues. An arm handles its later alarms live (detectors keep running inside the branch).

What an alarm does (arm kind):
  P0  nothing (the baseline only brings the others to their alarm).
  P1  re-roll: plan the alarm decision again with the noise key intervention_idx = "reroll"; no pause, no rewind; acts once.
  P2  blind rewind: pause 10 steps, arm-only rewind to the target, retry with fresh noise (intervention_idx + 1), at most N rewinds, then flag.
  P3  Second Look: pause 10 steps, DeltaDINO (model C) on (target frame, settled frame): CHANGED -> flag and stop, UNCHANGED -> as P2.
  P4  oracle gate: as P3 with the simulator truth (label_v2 between the target and the settled state).
  P6  check only: pause 10 steps, DeltaDINO; CHANGED -> flag, UNCHANGED -> continue the planned chunk without rewinding (the CUSUM is reset to 0).
  PA  (Part 3, SPEC_part3.md) Second Look with ONE grace: as P3, but the FIRST CHANGED verdict of the episode is not acted on: the CUSUM is reset to 0 and the planned chunk is executed
      (no rewind, as P6 after an UNCHANGED check); any alarm AFTER the grace is used flags at once (no pause, no gate). Before the first CHANGED verdict PA is bitwise P3.
  PAc as PA, but alarms after the grace go through the normal P3 path (pause, gate: CHANGED flags, UNCHANGED rewinds), the grace being used up.
  PS / PNr / PNd / PNa / PNp (Part 3 probe, src/second_look/probe.py, results/part3/notes/p3_probe_plan.txt): pause 10, gate; CHANGED: the grace of PA; UNCHANGED: stay (PS) or look around the alarm pose (PN*), NO rewind;
      any later alarm flags at once.
  SH  (Part 4, src/second_look/selective.py, SPEC_part4.md): pause 10, then retry (arm-only rewind to the target, ONE retry, fresh noise) or continue (CUSUM reset, execute the planned chunk), chosen by `mode`
      ("retry" = P2r, "continue" = PC, "model" = the retry-value model of the arm: retry iff p_retry - p_cont > delta; SH+F also flags when max p < tau). The gate is a FEATURE, never a rule.
      After the one intervention every later alarm is ignored (no second intervention, no flag).
P5 is P3 (or any kind) with trigger = "p5": its alarms are Bernoulli draws with the hazard measured on CAL.

Rewind = `snapshot.arm_only_restore` (arm and controller back to the target decision, objects stay); the budget clock never goes back: pause steps and the
return cost ((alarm decision - target decision) * 10 steps, 0 for the teleport run) are charged (second_look.budget). A rewind that lands the arm in penetration > 1 cm of
something is invalid and flags the episode. Frames for DeltaDINO come from a separate render Scene (second_look.gate.Frames): the policy's own simulator is never
rendered by us, so the extra renders cannot change what the policy sees (G4).
"""
from __future__ import annotations

import copy
import gc
import hashlib
from dataclasses import dataclass, field, replace

import numpy as np

from second_look import detectors, features, imgcheck, probe, selective, snapshot
from second_look.budget import LIMIT, PAUSE_STEPS, Budget, run_pause
from second_look.cusum import CUSUM, CusumParams
from second_look.policy import N_EXEC
from second_look.seeding import decision_noise, p5_uniform

RING = 12
TARGET_L = {"L1": 1, "L2": 2, "L4": 4, "L6": 6, "L10": 10, "LC4": 4}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str                       # "P0" | "P1" | "P2" | "P3" | "P4" | "P6" | "PA" | "PAc" (Part 3) | "PS" | "PNr" | "PNd" | "PNa" | "PNp" (Part 3 probe) | "SH" (Part 4)
    trigger: str = "accel"          # "accel" | "stac" | "p5"
    target: str = "L2"
    N: int = 1                      # rewinds allowed
    op: str = "F1-max"              # DeltaDINO operating point: key of RunCfg.thresholds
    seed: int = 0                   # DeltaDINO seed
    teleport: bool = False          # return cost 0
    mode: str = "model"             # SH (Part 4): "retry" | "continue" | "model"
    model: str = ""                 # SH: key of RunCfg.sh_models
    delta: float = 0.0              # SH: retry iff p_retry - p_cont > delta
    tau: float | None = None        # SH+F: flag iff max(p_retry, p_cont) < tau
    key: int = 0                    # SH: noise key of the branch: retry r = 1..4 (0 or 1: the standard retry key), continue c = 1..4 (0: the baseline keys)
    ns2: str = ""                   # Part 5 (P5.1b): after the first decision of the retry (planned with intervention index ns+key) every later decision uses f"{ns2}_{key}_{sub}"
    sub: int = 0
    ns: str = ""                    # Part 5: noise namespace "p5nc": the branch's intervention index is f"{ns}{key}" (retry mode "retry", or mode "cont_nopause": re-plan at the alarm, no pause, no rewind)


P0 = ArmSpec("P0", "P0")


@dataclass
class RunCfg:
    split: str
    task: int
    init: int
    seed_id: str
    accel: CusumParams
    accel_scale: np.ndarray                        # (70,)
    stac: CusumParams | None = None                # None: STAC is not run
    stac_k: int = 8
    thresholds: dict = field(default_factory=dict)       # (seed, op) or (seed, op, target) -> logit threshold of model C (gate: CHANGED when s >= threshold)
    p5_hazard: np.ndarray | None = None            # per decision index
    limit: int = LIMIT
    trace: bool = True                             # per-step state/action hashes (G1, G2)
    render_every_decision: bool = False            # G4: render a frame at every decision although nobody needs it
    memoise: bool = True                           # share the first pause + settled frame + DeltaDINO score between arms branched from the same alarm (bitwise-neutral, tested)
    keep_raw: bool = False                         # keep each decision's iterates / STAC resamples in the log (CAL, G6)
    p4: bool = False                               # Part 4: keep the end-effector position of every decision (feature: end-effector speed)
    sh_models: dict = field(default_factory=dict)  # Part 4: model name -> second_look.shmodel.SHModel
    probe_frames: bool = False                     # Part 3 probe: keep the policy env's camera frames at every probe pose in the log
    force_alarms: frozenset = frozenset()          # tests (G2, G5): decision indices that count as an alarm for every accel/STAC arm (every time they are reached)

    @property
    def reset_seed(self) -> int:
        return int.from_bytes(hashlib.sha256(f"reset|{self.split}|{self.task}|{self.init}".encode()).digest()[:4], "big") & 0x7FFFFFFF


# ------------------------------------------------------------------------------------------------ rig
class Rig:
    """Everything one worker owns: ONE policy env, the policy, the gate, the render Scene."""

    def __init__(self, device: str = "cuda"):
        from second_look import glfix
        glfix.patch()                                     # before any env exists
        from second_look.gate import DeltaGate, Frames
        from second_look.policy import SmolVLA
        self.vla = SmolVLA(device)
        self.gate = DeltaGate(device)
        self.frames = Frames()
        self.env = None
        self.task_idx = None

    def open_task(self, task: int):
        if self.task_idx == task and self.env is not None:
            return
        if self.env is not None:
            self._close_env()                             # destroy the old GL context now, in its own context
        from second_look import policy_rollout as PR
        self.env = PR.make_env(task, 0)
        self.task_idx = task

    @property
    def renv(self):
        return snapshot.robosuite_env(self.env)

    def _close_env(self):
        activate_env(self.renv)                           # freeing a render context frees GL objects in the CURRENT context: make it the env's own first
        self.env.close()
        self.env = None
        gc.collect()

    def close(self):
        if self.env is not None:
            self._close_env()
        self.frames.close()


def activate_env(renv) -> None:
    """Make the policy env's GL context current (our render Scene may have been the current one): robosuite renders into whatever is current."""
    ctx = getattr(renv.sim, "_render_context_offscreen", None)
    gl = getattr(ctx, "gl_ctx", None)
    if gl is not None and hasattr(gl, "make_current"):
        gl.make_current()


def get_flat(renv) -> np.ndarray:
    return renv.sim.get_state().flatten().copy()


def obs_hash(obs: dict) -> str:
    """Hash of everything the policy sees at a decision: both cameras and the robot state."""
    h = hashlib.blake2b(digest_size=8)

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o):
                walk(o[k])
        elif o is not None:
            h.update(np.ascontiguousarray(o).tobytes())
    walk(obs)
    return h.hexdigest()


def _h(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a, np.float64).tobytes(), digest_size=8).hexdigest()


# ------------------------------------------------------------------------------------------------ state
@dataclass
class Entry:
    d: int
    snap: dict
    flat: np.ndarray
    S_a: float
    S_s: float
    stac_prev: np.ndarray | None
    g_last: float
    elapsed: int
    cache: dict = field(default_factory=dict)             # lazily rendered frame


@dataclass
class Dec:
    actions: np.ndarray
    accel: float
    S_a: float
    alarm_a: bool
    stac: float | None = None
    S_s: float | None = None
    alarm_s: bool = False
    p5_u: float | None = None
    iterates: np.ndarray | None = None        # only with cfg.keep_raw (CAL, G6): (T+1, 50, 7) denoise path of the executed chunk
    samples: np.ndarray | None = None         # only with cfg.keep_raw and STAC: (K, 50, 7)


class St:
    """Mutable state of one run (baseline or an arm)."""

    def __init__(self, cfg: RunCfg):
        self.d = 0
        self.budget = Budget(limit=cfg.limit)
        self.iv: int | str = 0
        self.rewinds = 0
        self.cus_a = CUSUM(cfg.accel)
        self.cus_s = CUSUM(cfg.stac) if cfg.stac is not None else None
        self.stac_prev: np.ndarray | None = None
        self.g_last = -1.0
        self.obs: dict | None = None
        self.ring: list[Entry] = []
        self.done = False
        self.success = False
        self.flagged = False
        self.flag_reason: str | None = None
        self.rerolled = False
        self.grace_used = False                     # PA / PAc: the one grace (first CHANGED verdict) has been spent
        self.hist: list = []                        # Part 4: end-effector position at the start of every decision (only with cfg.p4)
        self.p5: dict = {}                          # Part 5 (arms with ns): first executed chunk after the intervention, end-of-branch label
        self.p5_ref = None                          # Part 5: flat state of the rewind target (reference of the end-of-branch label)
        self.alarm_done = False                     # probe kinds: the first alarm has been handled; any later alarm flags at once
        self.cut_short = False                      # a retry the budget could not pay for (return cost >= time left)
        self.log = {"decisions": [], "events": [], "trace": []}
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.invalid_rewinds = 0
        self.raw: list = []
        self.memo: dict | None = None               # shared by all arms branched from ONE alarm (run_branches); None = no memoisation
        self.plans: dict | None = None              # (iv, decision, observation hash) -> planned chunk; shared by EVERY run of the episode (copy.copy keeps the reference)

    def fork(self) -> "St":
        s = copy.copy(self)
        s.budget = copy.copy(self.budget)
        s.cus_a = copy.copy(self.cus_a)
        s.cus_s = copy.copy(self.cus_s) if self.cus_s is not None else None
        s.ring = list(self.ring)
        s.log = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in self.log.items()}
        s.pairs = list(self.pairs)
        s.hist = list(self.hist)
        s.raw = list(self.raw)
        s.p5 = {}
        return s


# ------------------------------------------------------------------------------------------------ pieces
def make_entry(rig: Rig, cfg: RunCfg, st: St) -> Entry:
    renv = rig.renv
    e = Entry(st.d, snapshot.capture(rig.env, meta={"decision": st.d}), get_flat(renv), st.cus_a.state, st.cus_s.state if st.cus_s else 0.0,
              st.stac_prev, st.g_last, st.budget.elapsed)
    if cfg.render_every_decision:
        e.cache["frame"] = rig.frames.render_flat(e.flat)
    return e


def decide(rig: Rig, cfg: RunCfg, st: St, need_stac: bool) -> Dec:
    imgcheck.check_obs(st.obs, f"{cfg.split} task {cfg.task} init {cfg.init} {cfg.seed_id} decision {st.d}")        # crash loudly on a corrupted render (src/second_look/glfix.py)
    # A plan is a deterministic function of (observation, noise key) (G1: bitwise). Arms that are still identical (same intervention index, decision and observation)
    # therefore share it: the profile shows planning (SmolVLA, 250 ms per decision) is two thirds of every arm's cost. Bitwise-neutral (tests/p2/test_live_memo.py).
    key = (st.iv, st.d, obs_hash(st.obs)) if (st.plans is not None and not cfg.keep_raw) else None
    hit = st.plans.get(key) if key is not None else None
    if hit is not None and need_stac and cfg.stac is not None and hit["samples"] is None:
        hit = None                                                  # cached by an arm that did not need the STAC resamples: plan again (same bits) and add them
    if hit is None:
        noise = decision_noise(cfg.split, cfg.task, cfg.init, cfg.seed_id, st.iv, st.d, ks=(0,))
        activate_env(rig.renv)
        plan, batch = rig.vla.plan(st.obs, rig.env.task_description, noise)
        hit = {"actions": plan.actions, "accel": detectors.accel(plan.iterates, cfg.accel_scale), "iterates": plan.iterates if cfg.keep_raw else None, "samples": None}
        if key is not None:
            st.plans[key] = hit
        batch_keep = batch
    else:
        batch_keep = None
    Sa, alarm_a = st.cus_a.update(hit["accel"])
    dec = Dec(hit["actions"], hit["accel"], Sa, alarm_a, iterates=hit["iterates"])
    if need_stac and cfg.stac is not None:
        if hit["samples"] is None:                                  # (batch_keep is the batch `plan` returned: always set here)
            noises = decision_noise(cfg.split, cfg.task, cfg.init, cfg.seed_id, st.iv, st.d, ks=range(1, cfg.stac_k + 1))
            activate_env(rig.renv)
            hit["samples"] = rig.vla.resample(batch_keep, noises)
        samples = hit["samples"]
        dec.stac = detectors.stac(samples, st.stac_prev)
        dec.S_s, dec.alarm_s = st.cus_s.update(dec.stac)
        st.stac_prev = samples
        dec.samples = samples                                       # (8, 50, 7): Part 4 features (within-pose spread); a reference, no copy
    if cfg.p5_hazard is not None:
        dec.p5_u = p5_uniform(cfg.split, cfg.task, cfg.init, cfg.seed_id, st.iv, st.d)
    return dec


def p5_alarm(cfg: RunCfg, dec: Dec, d: int) -> bool:
    if cfg.p5_hazard is None or dec.p5_u is None:
        return False
    h = cfg.p5_hazard
    return bool(dec.p5_u < float(h[min(d, len(h) - 1)]))


def alarm_for(cfg: RunCfg, trigger: str, dec: Dec, d: int) -> bool:
    forced = d in cfg.force_alarms
    return {"accel": dec.alarm_a or forced, "stac": dec.alarm_s or forced, "p5": p5_alarm(cfg, dec, d)}[trigger]


def env_step(rig: Rig, cfg: RunCfg, st: St, action: np.ndarray, tag: str) -> bool:
    """One physics/control step in the policy env. Returns task success after this step (checked after EVERY step)."""
    renv = rig.renv
    activate_env(renv)
    obs, _, terminated, _, info = rig.env.step(np.asarray(action, np.float64))
    st.obs = obs
    st.g_last = float(action[6])
    if cfg.trace:
        st.log["trace"].append((tag, _h(action), _h(get_flat(renv)) if not info.get("is_success", False) else "success"))
    return bool(info.get("is_success", False))


def pick_target(st: St, arm: ArmSpec) -> Entry:
    """Rewind target (SPEC §2.2): L decisions before the alarm decision, clamped at the oldest decision we have; LC4: the most recent decision within the last 4
    with CUSUM statistic S = 0 (after its update, i.e. the next decision's pre-state), else L4."""
    by = {e.d: e for e in st.ring}
    da = st.d
    want = da - TARGET_L[arm.target]
    if arm.target == "LC4":
        for j in range(da - 1, max(da - 4, 0) - 1, -1):
            nxt = by.get(j + 1)
            if j in by and nxt is not None and nxt.S_a == 0.0:
                want = j
                break
    want = max(want, 0)
    if want in by:
        return by[want]
    return min(st.ring, key=lambda e: abs(e.d - want))


def end(st: St, *, success=False, flag: str | None = None) -> None:
    st.done = True
    st.success = st.success or success
    if flag:
        st.flagged, st.flag_reason = True, flag


# ------------------------------------------------------------------------------------------------ alarm handling
def handle_alarm(rig: Rig, cfg: RunCfg, st: St, arm: ArmSpec, dec: Dec) -> tuple[str, Dec]:
    """Returns (what, dec): 'rewound' (loop again from the restored decision), 'continue' (execute `dec`), 'done' (the run is over)."""
    if arm.kind == "P1":
        if st.rerolled:
            return "continue", dec
        st.rerolled, st.iv = True, "reroll"
        noise = decision_noise(cfg.split, cfg.task, cfg.init, cfg.seed_id, "reroll", st.d, ks=(0,))
        activate_env(rig.renv)
        plan, _ = rig.vla.plan(st.obs, rig.env.task_description, noise)
        st.log["events"].append({"kind": "reroll", "d": st.d, "elapsed": st.budget.elapsed})
        return "continue", replace(dec, actions=plan.actions)
    if arm.kind == "SH" and st.alarm_done:                      # Part 4: after the one intervention every later alarm is ignored
        return "continue", dec
    if arm.kind == "SH" and arm.mode == "cont_nopause":         # Part 5: continue at the alarm with a fresh noise stream, no pause, no rewind
        return selective.handle_nopause(rig, cfg, st, arm, dec)
    if arm.kind in probe.PROBE_KINDS and st.alarm_done:         # a later alarm of a probe arm: flag at once (no pause, no second look)
        st.log["events"].append({"kind": "probe_flag", "d": st.d, "elapsed": st.budget.elapsed, "S_a": dec.S_a})
        end(st, flag="alarm_after_probe")
        return "done", dec
    if arm.kind == "PA" and st.grace_used:                     # an alarm after the grace: flag at once (no pause, no second look)
        st.log["events"].append({"kind": "grace_flag", "d": st.d, "elapsed": st.budget.elapsed, "S_a": dec.S_a})
        end(st, flag="alarm_after_grace")
        return "done", dec
    if arm.kind in ("P2", "P3", "P4", "PA", "PAc") and st.rewinds >= arm.N:
        st.log["events"].append({"kind": "retry_limit", "d": st.d, "elapsed": st.budget.elapsed})
        end(st, flag="retry_limit")
        return "done", dec
    # ---- pause. The FIRST pause of every arm branched from the same alarm starts from bitwise the same state with the same budget, so it is simulated once and replayed
    # from a full snapshot for the other arms (memo). Later pauses (after a rewind, or P6's second check) start from different states and are always live.
    memo = st.memo if (st.memo is not None and st.budget.paused == 0) else None
    pm = memo.get("pause") if memo is not None else None
    if pm is not None:
        for _ in range(pm["n"]):
            st.budget.charge_pause_step()
        st.log["trace"].extend(pm["trace"])
        st.g_last = pm["g_last"]
        st.obs = copy.deepcopy(pm["obs"])
        ok = pm["ok"]
        if not ok:
            activate_env(rig.renv)
            snapshot.restore(rig.env, pm["snap"])
    else:
        n0 = len(st.log["trace"])
        n_p, ok = run_pause(st.budget, lambda: env_step(rig, cfg, st, np.concatenate([np.zeros(6), [st.g_last]]), "pause"))
        if memo is not None:
            memo["pause"] = {"snap": None if ok else snapshot.capture(rig.env), "obs": copy.deepcopy(st.obs), "g_last": st.g_last, "n": n_p, "ok": ok, "trace": list(st.log["trace"][n0:])}
    if ok:
        end(st, success=True)
        return "done", dec
    if st.budget.exhausted:
        end(st)
        return "done", dec
    # ---- the second look
    target = pick_target(st, arm)
    renv = rig.renv
    flat_set = get_flat(renv)
    mm = memo if memo is not None else {}                      # per-alarm cache: only the first pause of an arm is shared (see above)

    def cached(key, fn):
        if memo is None:
            return fn()
        if key not in mm:
            mm[key] = fn()
        return mm[key]
    truth = cached(("truth", target.d), lambda: rig.frames.oracle(target.flat, flat_set))
    if cfg.memoise:
        ref_img = target.cache.get("frame")
        if ref_img is None:
            ref_img = target.cache["frame"] = rig.frames.render_flat(target.flat)
    else:
        ref_img = rig.frames.render_flat(target.flat)
    set_img = cached("settled", lambda: rig.frames.render_flat(flat_set))
    sc = cached(("score", target.d, arm.seed), lambda: rig.gate.score(ref_img, set_img, arm.seed))          # (score, 16x16 change map); measured for every pause arm (P2/P4 only log it)
    s = sc[0]
    thr = cfg.thresholds.get((arm.seed, arm.op, arm.target), cfg.thresholds.get((arm.seed, arm.op)))        # per-target thresholds (configs/part2_gate_cal.yaml), else one for all targets
    gate_changed = (s >= thr) if thr is not None else None
    if arm.kind == "P2":
        verdict = "UNCHANGED"                                  # blind: no check
    elif arm.kind == "P4":
        verdict = "CHANGED" if truth["label"] == 1 else "UNCHANGED"
    else:
        verdict = "CHANGED" if gate_changed else "UNCHANGED"
    st.pairs.append((ref_img, set_img))
    ev = {"kind": "gate", "d": st.d, "target_d": target.d, "gap_steps": (st.d - target.d) * N_EXEC + PAUSE_STEPS, "arm_kind": arm.kind, "trigger": arm.trigger,
          "s": s, "thr": thr, "gate_changed": gate_changed, "verdict": verdict, "truth": truth["label"], "truth_gray": truth["gray"], "truth_items": truth["changed_items"],
          "elapsed": st.budget.elapsed, "rewinds": st.rewinds, "pair": len(st.pairs) - 1, "S_a": dec.S_a, "iv": st.iv}
    st.log["events"].append(ev)
    if arm.kind == "SH":                                       # Part 4: the gate is a feature; retry or continue by the arm's mode / model
        return selective.handle(rig, cfg, st, arm, dec, ev, target, sc)
    if arm.kind in probe.PROBE_KINDS:
        if verdict == "CHANGED":                                # PA behaviour for ALL probe arms: the grace
            st.alarm_done = True
            ev["grace"] = True
            st.cus_a.restore(0.0)
            if st.cus_s is not None:
                st.cus_s.restore(0.0)
            return "continue", dec
        return probe.handle_unchanged(rig, cfg, st, arm, dec, memo)
    if verdict == "CHANGED" and arm.kind in ("PA", "PAc") and not st.grace_used:         # the grace: continue as P6 does after an UNCHANGED check
        st.grace_used = True
        ev["grace"] = True
        st.cus_a.restore(0.0)
        if st.cus_s is not None:
            st.cus_s.restore(0.0)
        return "continue", dec
    if verdict == "CHANGED":
        end(st, flag="changed")
        return "done", dec
    if arm.kind == "P6":
        st.cus_a.restore(0.0)
        if st.cus_s is not None:
            st.cus_s.restore(0.0)
        return "continue", dec
    # ---- rewind (arm only)
    st.budget.charge_return(0 if arm.teleport else (st.d - target.d) * N_EXEC)
    ev["return_cost"] = 0 if arm.teleport else (st.d - target.d) * N_EXEC
    if st.budget.exhausted:
        st.cut_short = True
        end(st)
        return "done", dec
    activate_env(renv)
    info = snapshot.arm_only_restore(rig.env, target.snap)
    ev.update(penetration=info["penetration"], valid=info["valid"])
    st.rewinds += 1
    if not info["valid"]:
        st.invalid_rewinds += 1
        end(st, flag="invalid_rewind")
        return "done", dec
    st.d = target.d
    st.iv = (st.iv + 1) if isinstance(st.iv, int) else st.iv
    st.cus_a.restore(target.S_a)
    if st.cus_s is not None:
        st.cus_s.restore(target.S_s)
    st.stac_prev = target.stac_prev
    st.g_last = target.g_last
    st.ring = [e for e in st.ring if e.d < target.d]
    st.obs = snapshot.observation(rig.env)
    return "rewound", dec


def execute(rig: Rig, cfg: RunCfg, st: St, dec: Dec) -> None:
    for a in dec.actions:
        if st.budget.exhausted:
            break
        st.budget.charge_step()
        if env_step(rig, cfg, st, a, "exec"):
            end(st, success=True)
            return
    st.d += 1


# ------------------------------------------------------------------------------------------------ the play loop
def play(rig: Rig, cfg: RunCfg, st: St, arm: ArmSpec, branch_arms: list[ArmSpec] | None = None, pending: Dec | None = None) -> dict:
    need_stac = cfg.stac is not None and (arm.kind == "P0" or arm.trigger == "stac")
    triggers = sorted({a.trigger for a in (branch_arms or [])})
    fired: set[str] = set()
    branches: dict[str, dict] = {}
    while not st.done:
        if st.budget.exhausted:
            end(st)
            break
        if pending is None:
            st.ring.append(make_entry(rig, cfg, st))
            st.ring = st.ring[-RING:]
            dec = decide(rig, cfg, st, need_stac)
            st.log["decisions"].append({"d": st.d, "iv": st.iv, "accel": dec.accel, "S_a": dec.S_a, "alarm_a": dec.alarm_a, "stac": dec.stac, "S_s": dec.S_s,
                                        "alarm_s": dec.alarm_s, "elapsed": st.budget.elapsed, "obs": obs_hash(st.obs)})
            if cfg.p4:
                st.hist.append(features.proprio(st.obs)[0])
            if cfg.keep_raw:
                st.raw.append((dec.iterates, dec.samples))
            if arm.ns and isinstance(st.iv, str) and "chunk" not in st.p5:
                st.p5.update(selective.p5_chunk(cfg, st, dec))
                if arm.ns2:
                    st.iv = f"{arm.ns2}_{arm.key}_{arm.sub}"
        else:
            dec, pending = pending, None
        if arm.kind == "P0":
            for trig in triggers:
                if trig not in fired and alarm_for(cfg, trig, dec, st.d):
                    fired.add(trig)
                    branches.update(run_branches(rig, cfg, st, dec, trig, [a for a in branch_arms if a.trigger == trig]))
        elif alarm_for(cfg, arm.trigger, dec, st.d):
            what, dec = handle_alarm(rig, cfg, st, arm, dec)
            if what == "done":
                break
            if what == "rewound":
                continue
        if st.done:
            break
        execute(rig, cfg, st, dec)
    if arm.ns and st.p5_ref is not None:
        st.p5["end_label"] = selective.end_label(rig, st)
    return {"success": bool(st.success), "flagged": st.flagged, "flag_reason": st.flag_reason, "elapsed": min(st.budget.elapsed, cfg.limit), "executed": st.budget.executed, "p5": st.p5,
            "cut_short": st.cut_short, "paused": st.budget.paused, "returned": st.budget.returned, "probed": st.budget.probed, "rewinds": st.rewinds, "invalid_rewinds": st.invalid_rewinds, "decisions": st.d,
            "log": st.log, "pairs": st.pairs, "raw": st.raw, "branches": branches}


def run_branches(rig: Rig, cfg: RunCfg, st: St, dec: Dec, trig: str, arms: list[ArmSpec]) -> dict[str, dict]:
    """Play each arm from a full restore of the snapshot at the current decision (the last ring entry), then restore the baseline to it."""
    entry = st.ring[-1]
    assert entry.d == st.d
    out = {}
    memo: dict | None = {} if cfg.memoise else None
    for arm in arms:
        activate_env(rig.renv)
        snapshot.restore(rig.env, entry.snap)
        st2 = st.fork()
        st2.memo = memo
        st2.obs = snapshot.observation(rig.env)
        out[arm.name] = play(rig, cfg, st2, arm, pending=dec)
    activate_env(rig.renv)
    snapshot.restore(rig.env, entry.snap)
    st.obs = snapshot.observation(rig.env)
    return out


def fresh_start(rig: Rig, cfg: RunCfg) -> St:
    rig.open_task(cfg.task)
    env = rig.env
    env.init_state_id = cfg.init
    obs, _ = env.reset(seed=cfg.reset_seed)
    renv = rig.renv
    rig.frames.bind(cfg.task, renv.sim.model.get_xml())
    st = St(cfg)
    st.obs = obs
    st.plans = {} if cfg.memoise else None
    return st


def run_episode(rig: Rig, cfg: RunCfg, arms: list[ArmSpec]) -> dict:
    """The baseline plus every arm of `arms`, paired. Returns {"P0": result, ...} (branch results are lifted to the top level)."""
    st = fresh_start(rig, cfg)
    res = play(rig, cfg, st, P0, branch_arms=arms)
    out = {"P0": {k: v for k, v in res.items() if k != "branches"}}
    out.update(res["branches"])
    return out


def run_arm_fresh(rig: Rig, cfg: RunCfg, arm: ArmSpec, at_decision: int) -> dict:
    """G2 reference: replay the baseline from reset (no snapshots restored, no branching) up to `at_decision`, plan it, then play `arm` from the LIVE state."""
    st = fresh_start(rig, cfg)
    need = arm.trigger == "stac" or (arm.kind == "SH" and cfg.stac is not None)          # SH needs the baseline's STAC / samples (features)
    while st.d < at_decision:
        st.ring.append(make_entry(rig, cfg, st))
        st.ring = st.ring[-RING:]
        dec = decide(rig, cfg, st, need)
        st.log["decisions"].append({"d": st.d, "iv": st.iv, "accel": dec.accel, "S_a": dec.S_a, "alarm_a": dec.alarm_a, "stac": dec.stac, "S_s": dec.S_s,
                                    "alarm_s": dec.alarm_s, "elapsed": st.budget.elapsed, "obs": obs_hash(st.obs)})
        if cfg.p4:
            st.hist.append(features.proprio(st.obs)[0])
        execute(rig, cfg, st, dec)
        assert not st.done, "episode ended before the requested decision"
    st.ring.append(make_entry(rig, cfg, st))
    st.ring = st.ring[-RING:]
    dec = decide(rig, cfg, st, need)
    assert at_decision in cfg.force_alarms, "run_arm_fresh needs the alarm to be forced at the branch decision"
    if arm.kind == "SH":                                                                  # the baseline logs the alarm decision before it branches: SH reads it
        st.log["decisions"].append({"d": st.d, "iv": st.iv, "accel": dec.accel, "S_a": dec.S_a, "alarm_a": dec.alarm_a, "stac": dec.stac, "S_s": dec.S_s,
                                    "alarm_s": dec.alarm_s, "elapsed": st.budget.elapsed, "obs": obs_hash(st.obs)})
        if cfg.p4:
            st.hist.append(features.proprio(st.obs)[0])
    return play(rig, cfg, st, arm, pending=dec)
