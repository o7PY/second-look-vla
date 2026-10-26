"""Part 4 arm kind SH (SPEC_part4.md sections 2 and 6): at the first alarm, after the 10-step pause (done by `second_look.loop.handle_alarm`, exactly as in Part 2) and the DeltaDINO score (a FEATURE, never a rule),
the arm RETRIES (arm-only rewind to the target, the return cost charged, ONE retry with fresh noise) or CONTINUES (CUSUM reset, the already planned chunk is executed), by `arm.mode`:
  "retry"     always retry               (arm P2r; GEN retry branch r)
  "continue"  always continue            (arm PC;  GEN continue branch c)
  "model"     retry iff p_retry - p_cont > delta, with (p_retry, p_cont) from `cfg.sh_models[arm.model]` (SH, SH-task, SH-noG); SH+F (arm.tau set) FLAGS instead when max(p_retry, p_cont) < tau.
After the one intervention every later alarm is ignored (handled in `second_look.loop.handle_alarm`): no second intervention and no flag, so S isolates the retry decision.
Noise keys: a retry uses intervention index `arm.key` (r = 1..4; 0 means the standard retry key 1), a continue branch c >= 1 uses 2000 + c for every decision AFTER the alarm decision (c = 0: the baseline keys).
Every SH arm logs one event "sh" with the feature vector, (p_retry, p_cont) and the decision.
"""
from __future__ import annotations

import numpy as np

from second_look import features as F
from second_look import snapshot
from second_look.policy import N_EXEC

CONTINUE_KEY0 = 2000


def build_inputs(rig, cfg, st, dec, target, sc) -> F.FeatureInputs:
    from second_look import loop as L
    s, z = sc
    dl = st.log["decisions"]
    assert dl[-1]["d"] == st.d, "the decision log must end at the alarm decision"
    assert dec.samples is not None, "the alarm decision's samples are needed (STAC must be on for the P0 baseline: cfg.stac)"
    eef, grip = F.proprio(st.obs)
    h = st.hist
    d = st.d
    assert len(h) >= d + 1, "cfg.p4 must be on (end-effector history)"
    static = rig.gate.model(0).static.detach().cpu().numpy().astype(bool)
    return F.FeatureInputs(task=cfg.task, d=d, elapsed=st.budget.elapsed, limit=cfg.limit, return_cost=(d - target.d) * N_EXEC, accel_hist=[x["accel"] for x in dl], cusum_hist=[x["S_a"] for x in dl],
                           stac_alarm=float(dec.stac or 0.0), samples=dec.samples, gate_score=float(s), gate_map=np.asarray(z), static_mask=static, gripper_qpos=grip, eef_pos=eef,
                           eef_pos_prev=h[d - 1] if d >= 1 else h[d], eef_pos_alarm=h[d], accel_mu0=cfg.accel.mu0, accel_sigma=cfg.accel.sigma)


def _iv(arm):
    return f"{arm.ns}{arm.key}"


def p5_chunk(cfg, st, dec) -> dict:
    """Part 5: what a DSRL actor would need later: the noise seed / shape / dtype of the first decision planned with the fresh stream, and the chunk it executes."""
    from second_look.seeding import noise_seed
    return {"d": st.d, "iv": st.iv, "noise_seed": noise_seed(cfg.split, cfg.task, cfg.init, cfg.seed_id, st.iv, st.d, 0), "noise_shape": [1, 50, 32], "noise_dtype": "float32",
            "chunk": np.round(np.asarray(dec.actions, np.float64), 5).tolist()}


def end_label(rig, st) -> dict:
    """Part 5: label_v2 of the scene at the end of a branch against the reference before the failure (the rewind target, L6)."""
    from second_look import loop as L
    t = rig.frames.oracle(st.p5_ref, L.get_flat(rig.renv))
    return {"label": t["label"], "gray": t["gray"], "changed_items": t["changed_items"]}


def handle_nopause(rig, cfg, st, arm, dec):
    """Part 5 mode C: no pause, no rewind; the alarm decision is re-planned with the fresh stream and every later decision uses it; later alarms are ignored."""
    from dataclasses import replace
    from second_look import loop as L
    from second_look.seeding import decision_noise
    st.p5_ref = L.pick_target(st, arm).flat
    st.alarm_done = True
    st.iv = _iv(arm)
    st.log["events"].append({"kind": "p5_continue", "d": st.d, "iv": st.iv, "elapsed": st.budget.elapsed})
    st.cus_a.restore(0.0)
    if st.cus_s is not None:
        st.cus_s.restore(0.0)
    noise = decision_noise(cfg.split, cfg.task, cfg.init, cfg.seed_id, st.iv, st.d, ks=(0,))
    L.activate_env(rig.renv)
    plan, _ = rig.vla.plan(st.obs, rig.env.task_description, noise)
    new = replace(dec, actions=plan.actions)
    st.p5.update(p5_chunk(cfg, st, new))
    return "continue", new


def handle(rig, cfg, st, arm, dec, ev, target, sc):
    """Returns (what, dec) as `second_look.loop.handle_alarm`."""
    from second_look import loop as L
    x = F.features(build_inputs(rig, cfg, st, dec, target, sc))
    p_r = p_c = None
    flag = False
    if arm.mode == "model":
        p_r, p_c = cfg.sh_models[arm.model].predict(x)
        flag = arm.tau is not None and max(p_r, p_c) < arm.tau
        retry = (p_r - p_c) > arm.delta
    else:
        retry = arm.mode == "retry"
    st.log["events"].append({"kind": "sh", "d": st.d, "arm_kind": "SH", "mode": arm.mode, "model": arm.model, "delta": arm.delta, "tau": arm.tau, "key": arm.key, "p_retry": p_r, "p_cont": p_c,
                             "decision": "flag" if flag else ("retry" if retry else "continue"), "elapsed": st.budget.elapsed, "target_d": target.d, "x": x.tolist()})
    st.alarm_done = True
    st.p5_ref = target.flat
    if flag:
        L.end(st, flag="low_value")
        return "done", dec
    if not retry:
        st.cus_a.restore(0.0)
        if st.cus_s is not None:
            st.cus_s.restore(0.0)
        if arm.ns:
            st.iv = _iv(arm)
        elif arm.key:
            st.iv = CONTINUE_KEY0 + arm.key
        return "continue", dec
    # ---- retry: arm-only rewind to the target, return cost charged, one retry with fresh noise (as Part 2's P2)
    cost = (st.d - target.d) * N_EXEC
    st.budget.charge_return(cost)
    ev["return_cost"] = cost
    if st.budget.exhausted:
        st.cut_short = True
        L.end(st)
        return "done", dec
    L.activate_env(rig.renv)
    info = snapshot.arm_only_restore(rig.env, target.snap)
    ev.update(penetration=info["penetration"], valid=info["valid"])
    st.rewinds += 1
    if not info["valid"]:
        st.invalid_rewinds += 1
        L.end(st, flag="invalid_rewind")
        return "done", dec
    st.d = target.d
    st.iv = _iv(arm) if arm.ns else (arm.key if arm.key else 1)
    st.cus_a.restore(target.S_a)
    if st.cus_s is not None:
        st.cus_s.restore(target.S_s)
    st.stac_prev = target.stac_prev
    st.g_last = target.g_last
    st.ring = [e for e in st.ring if e.d < target.d]
    st.obs = snapshot.observation(rig.env)
    return "rewound", dec
