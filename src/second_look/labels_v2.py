"""label_v2: rotation of rotationally symmetric objects is measured as TILT (angle between the object's own up axis before and after);
spin about the up axis counts as zero. Everything else (translation, joints, gray zone, thresholds) is exactly as in v1 (SPEC §3.3)."""

from __future__ import annotations

import numpy as np

from second_look.scene import Delta, Item, R_CHG, R_GRAY, T_CHG, T_GRAY, quat_angle_deg

SYMMETRIC = ("akita_black_bowl_1_main", "plate_1_main", "wine_bottle_1_main")     # decided from renders: results/part1/p15/notes/c3_symmetry.md
UP_LOCAL = np.array([0.0, 0.0, 1.0])                                                 # every free object's own up axis is local +z


def up_vector(q: np.ndarray) -> np.ndarray:
    """World direction of the object's local +z for a (w, x, y, z) quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def tilt_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(up_vector(q0), up_vector(q1)), -1.0, 1.0))))


def compare_v2(registry: list[Item], r0: dict, r1: dict, symmetric: tuple[str, ...] = SYMMETRIC) -> list[Delta]:
    out = []
    for it in registry:
        a, b = r0[it.name], r1[it.name]
        if it.kind == "free":
            tr = float(np.linalg.norm(b[:3] - a[:3]))
            ro = tilt_deg(a[3:7], b[3:7]) if it.name in symmetric else quat_angle_deg(a[3:7] / np.linalg.norm(a[3:7]), b[3:7] / np.linalg.norm(b[3:7]))
            chg = tr > T_CHG or ro > R_CHG
            gray = (not chg) and (tr >= T_GRAY or ro >= R_GRAY)
        elif it.kind == "slide":
            tr, ro = float(abs(b[0] - a[0])), 0.0
            chg, gray = tr > T_CHG, (T_GRAY <= tr <= T_CHG)
        else:
            tr, ro = 0.0, float(np.degrees(abs(b[0] - a[0])))
            chg, gray = ro > R_CHG, (R_GRAY <= ro <= R_CHG)
        out.append(Delta(it.name, it.kind, tr, ro, "changed" if chg else ("gray" if gray else "same")))
    return out


def label_pair_v2(scene, r_ref: dict, r_set: dict) -> dict:
    """Same return value as ``second_look.scene.label_pair`` but with the v2 rotation rule (symmetric objects: tilt only). Used to build new data directly with v2 labels."""
    from second_look.scene import label_from_deltas
    deltas = compare_v2(scene.registry, r_ref, r_set)
    y, gray = label_from_deltas(deltas)
    return {"label": y, "gray": gray, "deltas": deltas, "changed_items": [d.name for d in deltas if d.status == "changed"],
            "max_trans": max((d.trans for d in deltas), default=0.0), "max_rot": max((d.rot for d in deltas), default=0.0)}
