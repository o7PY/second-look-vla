"""G7: the oracle label between two qpos vectors equals Part 1's labeller (label_pair_v2 on Scene readings) on 50 hand-built cases (objects teleported),
and equals the answer fixed by construction. Live: needs a Scene. P2_LIVE=1 python -m pytest tests/p2/test_live_oracle.py"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from second_look.labels_v2 import SYMMETRIC, label_pair_v2
from second_look.oracle import oracle_label
from livehelp import scene_for

pytestmark = pytest.mark.live
TASK = "open_the_middle_drawer_of_the_cabinet"


def _wxyz_mul(q_world_rot_xyzw, q_wxyz):
    q = R.from_quat(np.r_[q_wxyz[1:], q_wxyz[0]])
    out = (R.from_quat(q_world_rot_xyzw) * q).as_quat()
    return np.r_[out[3], out[:3]]


# (kind, magnitude, expected label, expected gray)
CASES = [("none", 0.0, 0, False), ("move", 0.003, 0, False), ("move", 0.007, 0, True), ("move", 0.015, 1, False), ("move", 0.08, 1, False),
         ("rot_z", 10.0, 1, False), ("rot_z", 3.5, 0, True), ("spin_sym", 40.0, 0, False), ("tilt_sym", 10.0, 1, False), ("tilt_sym", 3.5, 0, True),
         ("slide", 0.02, 1, False), ("slide", 0.007, 0, True)]


def test_g7_oracle_equals_part1_labeller_on_50_cases():
    rng = np.random.default_rng(0)
    with scene_for(TASK) as (sc, S, A):
        sc.set_flat(S[0])
        base = sc.get_flat()
        nq = sc.m.nq
        free = [it for it in sc.registry if it.kind == "free"]
        sym = [it for it in free if it.name in SYMMETRIC]
        non_sym = [it for it in free if it.name not in SYMMETRIC]
        slides = [it for it in sc.registry if it.kind == "slide"]
        assert sym and non_sym and slides, "the task must have symmetric, non-symmetric and sliding items"
        n_done = 0
        for rep in range(5):
            for kind, mag, want, want_gray in CASES:
                if n_done == 50:
                    break
                flat = base.copy()
                q = flat[1:1 + nq]                                              # view into qpos
                if kind == "move":
                    it = free[rng.integers(len(free))]
                    ang = rng.uniform(0, 2 * np.pi)
                    q[it.qadr:it.qadr + 2] += mag * np.array([np.cos(ang), np.sin(ang)])
                elif kind == "rot_z":
                    it = non_sym[rng.integers(len(non_sym))]
                    q[it.qadr + 3:it.qadr + 7] = _wxyz_mul(R.from_euler("z", mag, degrees=True).as_quat(), q[it.qadr + 3:it.qadr + 7])
                elif kind == "spin_sym":
                    it = sym[rng.integers(len(sym))]
                    q[it.qadr + 3:it.qadr + 7] = _wxyz_mul(R.from_euler("z", mag, degrees=True).as_quat(), q[it.qadr + 3:it.qadr + 7])
                elif kind == "tilt_sym":
                    it = sym[rng.integers(len(sym))]
                    q[it.qadr + 3:it.qadr + 7] = _wxyz_mul(R.from_euler("x", mag, degrees=True).as_quat(), q[it.qadr + 3:it.qadr + 7])
                elif kind == "slide":
                    it = slides[rng.integers(len(slides))]
                    q[it.qadr] += mag
                # Part 1's route: restore both states in the Scene, read, label with label_pair_v2
                sc.set_flat(base); r0 = sc.read()
                sc.set_flat(flat); r1 = sc.read()
                part1 = label_pair_v2(sc, r0, r1)
                ours = oracle_label(sc.registry, base[1:1 + nq], flat[1:1 + nq])
                assert (ours["label"], ours["gray"], ours["changed_items"]) == (part1["label"], part1["gray"], part1["changed_items"]), (rep, kind, mag)
                assert abs(ours["max_trans"] - part1["max_trans"]) < 1e-12 and abs(ours["max_rot"] - part1["max_rot"]) < 1e-9
                assert (ours["label"], ours["gray"]) == (want, want_gray), f"by construction: {kind} {mag}: got {ours}"
                n_done += 1
        assert n_done == 50
