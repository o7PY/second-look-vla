"""Part 4 features: non-live unit tests (no GPU, no env)."""
import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from second_look import features as F

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_INPUTS = {"task", "d", "elapsed", "limit", "return_cost", "accel_hist", "cusum_hist", "stac_alarm", "samples", "gate_score", "gate_map", "static_mask", "gripper_qpos", "eef_pos", "eef_pos_prev",
                  "eef_pos_alarm", "accel_mu0", "accel_sigma", "meta"}


def inputs(**kw):
    r = np.random.default_rng(0)
    base = dict(task=3, d=9, elapsed=100, limit=300, return_cost=60, accel_hist=list(np.linspace(0.5, 1.2, 10)), cusum_hist=[0, 0, 0.1, 0.3, 0.2, 0.5, 0.9, 1.0, 1.1, 1.3], stac_alarm=0.31,
                samples=r.normal(size=(8, 50, 7)).astype(np.float32), gate_score=4.2, gate_map=r.normal(size=(16, 16)), static_mask=np.eye(16, dtype=bool), gripper_qpos=np.array([0.02, -0.02]),
                eef_pos=np.array([0.1, 0.2, 0.3]), eef_pos_prev=np.array([0.1, 0.2, 0.28]), eef_pos_alarm=np.array([0.1, 0.2, 0.3]), accel_mu0=0.65, accel_sigma=0.22)
    base.update(kw)
    return F.FeatureInputs(**base)


def test_the_inputs_are_the_allowed_ones_and_none_is_simulator_truth():
    names = {f.name for f in dataclasses.fields(F.FeatureInputs)}
    assert names == ALLOWED_INPUTS, names ^ ALLOWED_INPUTS
    for n in names:
        assert not any(w in n.lower() for w in ("truth", "label", "oracle", "flat", "object", "qpos_sim", "sim")), n


def test_the_module_does_not_touch_the_simulator():
    src = (ROOT / "src" / "second_look" / "features.py").read_text()
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "dataclasses", "numpy"}, imported
    idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    for bad in ("mujoco", "Scene", "oracle", "label_v2", "set_flat", "snapshot", "renv", "sim", "env", "registry", "truth"):
        assert not any(bad.lower() == i.lower() or i.lower().startswith(bad.lower() + "_") for i in idents), (bad, idents)


def test_vector_shape_names_groups_and_determinism():
    x = F.features(inputs())
    assert x.shape == (len(F.FEATURE_NAMES),) == (28,) and np.isfinite(x).all()
    assert np.array_equal(x, F.features(inputs()))
    assert x[F.FEATURE_NAMES.index("task_3")] == 1.0 and x[:10].sum() == 1.0
    assert F.columns("T") == list(range(10)) and set(F.columns("TADGR")) == set(range(28))
    assert not set(F.columns("TADR")) & set(F.columns("G")) and len(F.columns("G")) == 4


def test_feature_values():
    x = dict(zip(F.FEATURE_NAMES, F.features(inputs())))
    assert x["alarm_decision"] == 9 and x["steps_used"] == 100 and x["steps_left_after_retry"] == 140
    assert np.isclose(x["cusum"], 1.3)
    assert np.isclose(x["cusum_slope3"], (1.3 - 0.9) / 3)
    assert np.isclose(x["accel_last4"], np.mean(np.linspace(0.5, 1.2, 10)[-4:]))
    assert x["n_cusum_positive"] == 8
    assert np.isclose(x["gripper_opening"], 0.04) and x["gripper_closed"] in (0.0, 1.0)
    assert np.isclose(x["ee_speed"], 0.002)
    top = np.sort(inputs().gate_map[~np.eye(16, dtype=bool)])[::-1][:3]
    assert np.allclose([x["gate_top1"], x["gate_top2"], x["gate_top3"]], top)


def test_within_spread_matches_part_3_definition():
    s = np.random.default_rng(5).normal(size=(8, 50, 7))
    ref = np.mean([np.linalg.norm(s[i, :10, :6] - s[j, :10, :6], axis=-1).mean() for i in range(8) for j in range(i + 1, 8)])
    assert np.isclose(F.within_spread(s), ref)


def test_proprio_reads_the_policy_observation_only():
    obs = {"pixels": {}, "robot_state": {"eef": {"pos": np.array([1.0, 2.0, 3.0])}, "gripper": {"qpos": np.array([0.03, -0.03])}}}
    p, g = F.proprio(obs)
    assert p.tolist() == [1.0, 2.0, 3.0] and g.tolist() == [0.03, -0.03]
