"""Part 4 features (SPEC_part4.md section 3): everything the retry-value model may see at the first alarm, after the 10-step pause. NO simulator truth, ever.

The only inputs are the fields of `FeatureInputs` (a guard test enumerates them and AST-checks this module: no simulator, scene or oracle names). Groups:
  T  task one-hot (10)
  A  alarm: alarm decision index, steps used, steps left after the retry cost, CUSUM, CUSUM slope over the last 3 decisions, accel over the last 4 decisions (raw, z-scored with the frozen CAL mu0 / sigma),
     number of decisions with CUSUM > 0
  D  disagreement at the alarm decision: STAC (the detector's own value) and the within-pose spread D_w of the 8 samples planned at that decision (Part 3 definition: mean pairwise per-step L2 over the 6 pose
     dims, executed window, normalised action space). Both come from the alarm decision's own 8 samples (the STAC detector computes them anyway): no extra plan, no extra charge.
  G  gate: DeltaDINO score for (L6 reference, settled frame after the pause) and the top-3 patch change logits among the non-static patches
  R  robot (policy observation after the pause): gripper opening, closed flag, end-effector speed over the last decision, end-effector height
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_TASKS = 10
EXEC_WINDOW = 10
CLOSED_BELOW = 0.04            # gripper opening (finger qpos difference, metres) below which the gripper counts as closed (holding or squeezing); fixed a priori, not tuned
N_EXEC = 10

GROUPS = {
    "T": [f"task_{i}" for i in range(N_TASKS)],
    "A": ["alarm_decision", "steps_used", "steps_left_after_retry", "cusum", "cusum_slope3", "accel_last4", "accel_last4_z", "n_cusum_positive"],
    "D": ["stac_alarm", "within_spread"],
    "G": ["gate_score", "gate_top1", "gate_top2", "gate_top3"],
    "R": ["gripper_opening", "gripper_closed", "ee_speed", "ee_height"],
}
FEATURE_NAMES = [n for g in "TADGR" for n in GROUPS[g]]
GROUP_OF = {n: g for g, ns in GROUPS.items() for n in ns}


def columns(groups: str) -> list[int]:
    return [i for i, n in enumerate(FEATURE_NAMES) if GROUP_OF[n] in groups]


@dataclass
class FeatureInputs:
    task: int
    d: int                                   # alarm decision index
    elapsed: int                             # steps used (after the pause)
    limit: int
    return_cost: int                         # steps a retry from here would cost to return (alarm decision - target decision) x 10
    accel_hist: list[float]                  # accel of decisions 0..d (the detector's own values)
    cusum_hist: list[float]                  # CUSUM statistic after decisions 0..d
    stac_alarm: float
    samples: np.ndarray                      # (8, 50, 7) the samples planned at the alarm decision
    gate_score: float
    gate_map: np.ndarray                     # (16, 16) DeltaDINO change-map logits
    static_mask: np.ndarray                  # (16, 16) bool, True = patch masked in the gate (never changes)
    gripper_qpos: np.ndarray                 # (2,) after the pause
    eef_pos: np.ndarray                      # (3,) after the pause
    eef_pos_prev: np.ndarray                 # (3,) at the start of the previous decision
    eef_pos_alarm: np.ndarray                # (3,) at the start of the alarm decision
    accel_mu0: float
    accel_sigma: float
    meta: dict = field(default_factory=dict)  # never read by `features`


def proprio(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """(end-effector position (3,), gripper finger positions (2,)) from the policy observation dict (the LIBERO env's `robot_state`)."""
    rs = obs["robot_state"]
    return np.asarray(rs["eef"]["pos"], np.float64).copy(), np.asarray(rs["gripper"]["qpos"], np.float64).copy()


def within_spread(samples: np.ndarray) -> float:
    x = np.asarray(samples, np.float64)[:, :EXEC_WINDOW, :6]
    iu = np.triu_indices(x.shape[0], 1)
    return float(np.linalg.norm(x[iu[0]] - x[iu[1]], axis=-1).mean())


def features(i: FeatureInputs) -> np.ndarray:
    """The feature vector in FEATURE_NAMES order (float64)."""
    f = {n: 0.0 for n in FEATURE_NAMES}
    f[f"task_{int(i.task)}"] = 1.0
    S = np.asarray(i.cusum_hist, np.float64)
    a = np.asarray(i.accel_hist, np.float64)
    last4 = a[-4:]
    f["alarm_decision"] = float(i.d)
    f["steps_used"] = float(i.elapsed)
    f["steps_left_after_retry"] = float(i.limit - i.elapsed - i.return_cost)
    f["cusum"] = float(S[-1])
    f["cusum_slope3"] = float((S[-1] - S[max(0, len(S) - 4)]) / max(1, min(3, len(S) - 1)))
    f["accel_last4"] = float(last4.mean())
    f["accel_last4_z"] = float((last4.mean() - i.accel_mu0) / i.accel_sigma)
    f["n_cusum_positive"] = float((S > 0).sum())
    f["stac_alarm"] = float(i.stac_alarm)
    f["within_spread"] = within_spread(i.samples)
    z = np.asarray(i.gate_map, np.float64)[~np.asarray(i.static_mask, bool)]
    top = np.sort(z)[::-1][:3]
    f["gate_score"] = float(i.gate_score)
    for k in range(3):
        f[f"gate_top{k + 1}"] = float(top[k])
    q = np.asarray(i.gripper_qpos, np.float64)
    opening = float(abs(q[0] - q[1]))
    f["gripper_opening"] = opening
    f["gripper_closed"] = float(opening < CLOSED_BELOW)
    f["ee_speed"] = float(np.linalg.norm(np.asarray(i.eef_pos_alarm) - np.asarray(i.eef_pos_prev)) / N_EXEC)
    f["ee_height"] = float(np.asarray(i.eef_pos)[2])
    return np.array([f[n] for n in FEATURE_NAMES], np.float64)
