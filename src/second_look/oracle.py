"""Simulator-truth change label between two moments (the P4 oracle gate, G7). Pure function of two `qpos` vectors; no Scene, no rendering.

It is `second_look.labels_v2.label_pair_v2` (the labeller Part 1 built its labels with) applied to the graded readings of `second_look.scene.Scene.read`
(free object -> [x y z qw qx qy qz]; slide / hinge -> [q]) taken from the qpos of the two states. A gray pair (a change between 0.5 and 1 cm / 2.5 and 5 deg,
no larger one) is label 0 with gray = True: the oracle gate calls it "unchanged" and the log keeps the gray flag.
"""
from __future__ import annotations

import numpy as np

from second_look.labels_v2 import compare_v2
from second_look.scene import Item, label_from_deltas


def readings(registry: list[Item], qpos: np.ndarray) -> dict[str, np.ndarray]:
    return {it.name: np.array(qpos[it.qadr:it.qadr + (7 if it.kind == "free" else 1)], dtype=np.float64) for it in registry}


def oracle_label(registry: list[Item], qpos_ref: np.ndarray, qpos_set: np.ndarray) -> dict:
    deltas = compare_v2(registry, readings(registry, qpos_ref), readings(registry, qpos_set))
    y, gray = label_from_deltas(deltas)
    return {"label": int(y), "gray": bool(gray), "changed_items": [d.name for d in deltas if d.status == "changed"],
            "max_trans": max((d.trans for d in deltas), default=0.0), "max_rot": max((d.rot for d in deltas), default=0.0)}
