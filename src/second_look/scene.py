"""LIBERO-Goal scene access: env construction, demo-state restore, object/joint registry, labels.

Ground truth only. Nothing in this module is ever a model input; it produces the answer key
(SPEC §3.3) and the masks (§3.4).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

SUITE = "libero_goal"
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
CAMERA = "agentview"

# Thresholds (SPEC §3.3). Translation in metres, rotation in degrees.
T_CHG, T_GRAY = 0.010, 0.005          # free objects & drawers: changed above T_CHG, gray in [T_GRAY, T_CHG]
R_CHG, R_GRAY = 5.0, 2.5              # free objects & hinges (degrees)

ROOT = Path(os.environ.get("SL_ROOT", Path(__file__).resolve().parent.parent.parent))
DEMO_DIR = ROOT / "data" / "raw" / "libero_goal"


def is_robot_name(name: str | None) -> bool:
    return bool(name) and name.startswith(ROBOT_PREFIXES)


def task_names() -> list[str]:
    from libero.libero import benchmark
    return list(benchmark.get_benchmark_dict()[SUITE]().get_task_names())


def demo_path(task_name: str) -> Path:
    return DEMO_DIR / f"{task_name}_demo.hdf5"


def fix_xml_paths(xml: str) -> str:
    """Demo XMLs carry the authors' absolute paths. Re-point ``assets/...`` to the project-local LIBERO
    assets and ``robosuite/...`` to the installed robosuite."""
    import robosuite
    from libero.libero import get_libero_path
    assets = get_libero_path("assets")
    rs = os.path.dirname(robosuite.__file__)

    def sub(m: re.Match) -> str:
        p = m.group(1)
        if "/robosuite/" in p:          # checked first: robosuite paths also contain "/assets/"
            return f'file="{rs}/{p.rsplit("/robosuite/", 1)[1]}"'
        if "/assets/" in p:
            return f'file="{assets}/{p.rsplit("/assets/", 1)[1]}"'
        return m.group(0)

    return re.sub(r'file="([^"]*)"', sub, xml)


@dataclass
class Item:
    """One thing whose change is graded: a free object or a non-robot articulated joint."""
    name: str                       # body name (free) or joint name (hinge/slide)
    kind: str                       # "free" | "slide" | "hinge"
    jnt: int
    body: int                       # body carrying the joint
    qadr: int
    subtree: np.ndarray = field(repr=False, default=None)   # body ids moving with this item


class Scene:
    """One LIBERO-Goal task env. Restores demo states and reads graded state."""

    def __init__(self, task_idx: int, render_px: int = 448):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from second_look import render
        render.patch()                  # MSAA off before any render context exists
        suite = benchmark.get_benchmark_dict()[SUITE]()
        self.task_idx = task_idx
        self.task_name = suite.get_task_names()[task_idx]
        self.bddl = suite.get_task_bddl_file_path(task_idx)
        self._OffScreenRenderEnv = OffScreenRenderEnv
        self.env = OffScreenRenderEnv(bddl_file_name=self.bddl, camera_heights=render_px,
                                      camera_widths=render_px, use_camera_obs=False,
                                      ignore_done=True, horizon=10**6)
        self.env.reset()
        self._bind()
        self.registry: list[Item] = []
        self._build_registry()

    # ---- model binding --------------------------------------------------------------------
    @property
    def renv(self):
        return self.env.env

    @property
    def sim(self):
        return self.renv.sim

    @property
    def m(self) -> mujoco.MjModel:
        return self.sim.model._model

    @property
    def d(self) -> mujoco.MjData:
        return self.sim.data._data

    def _bind(self) -> None:
        self.body_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.m.nbody)]

    def load_demo_model(self, xml: str) -> None:
        """Load the XML a demo was recorded with (carries that episode's fixture layout)."""
        self.renv.reset_from_xml_string(fix_xml_paths(xml))
        self._bind()
        self._build_registry()

    # ---- state ----------------------------------------------------------------------------
    def get_flat(self) -> np.ndarray:
        return self.sim.get_state().flatten().copy()

    def set_flat(self, flat: np.ndarray) -> None:
        self.sim.set_state_from_flattened(np.asarray(flat, dtype=np.float64))
        self.sim.forward()
        self.sync_visual()

    # ---- render state that is NOT in the physics state (R3) ----------------------------------------------------
    def sync_visual(self) -> None:
        """Recompute LIBERO's derived visual state from the (just restored) physics state.

        LIBERO shows the stove burner glow by toggling ``site_rgba[burner][3]`` (alpha -1 hidden, 2 shown, via ``1 - alpha``) from a Python flag that
        ``FlatStove.turn_on(qpos)`` sets only while a task predicate is evaluated. A plain state restore touches neither, so the glow left over from
        whatever was stepped last stayed on screen (stale glow, ~11-15% of the unchanged stove-task pairs of Part 1). Here the flag is set with LIBERO's own
        rule and the alpha is written explicitly, so a render is a function of the stored state. Idempotent."""
        renv = self.renv
        names = getattr(renv, "visualization_sites_list", [])
        joint_by_name = {it.name: it for it in self.registry}
        for name in dict.fromkeys(names):
            obj = renv.get_object(name)
            if not hasattr(obj, "turn_on"):
                continue                                           # only the stove has a state-dependent visual site in LIBERO-Goal (see results/part1/b2/notes/r3_render_state.md)
            knob = joint_by_name.get(f"{name}_button")
            if knob is None:
                continue
            visible = bool(obj.turn_on(float(self.d.qpos[knob.qadr])))          # sets obj.object_properties["vis_site_names"]["burner"] too
            for _, (site_name, _flag) in obj.object_properties["vis_site_names"].items():
                sid = self.sim.model.site_name2id(site_name)
                a = float(self.sim.model.site_rgba[sid][3])
                hidden = a if a <= 0 else 1.0 - a                 # LIBERO toggles alpha as 1 - alpha
                self.sim.model.site_rgba[sid][3] = (1.0 - hidden) if visible else hidden

    # ---- registry -------------------------------------------------------------------------
    def _subtree(self, body: int) -> np.ndarray:
        par = self.m.body_parentid
        out = []
        for b in range(self.m.nbody):
            x = b
            while True:
                if x == body:
                    out.append(b)
                    break
                if x == 0:
                    break
                x = par[x]
        return np.array(out, dtype=np.int32)

    def _build_registry(self) -> None:
        m = self.m
        items: list[Item] = []
        for j in range(m.njnt):
            t = int(m.jnt_type[j])
            body = int(m.jnt_bodyid[j])
            bname = self.body_names[body]
            if is_robot_name(bname):
                continue
            if t == mujoco.mjtJoint.mjJNT_FREE:
                kind = "free"
            elif t == mujoco.mjtJoint.mjJNT_SLIDE:
                kind = "slide"
            elif t == mujoco.mjtJoint.mjJNT_HINGE:
                kind = "hinge"
            else:
                continue
            name = bname if kind == "free" else (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}")
            items.append(Item(name, kind, j, body, int(m.jnt_qposadr[j]), self._subtree(body)))
        self.registry = items

    def registry_markdown(self) -> str:
        rows = ["| # | name | kind | body | subtree bodies |", "|---|---|---|---|---|"]
        for i, it in enumerate(self.registry):
            rows.append(f"| {i} | `{it.name}` | {it.kind} | `{self.body_names[it.body]}` | {len(it.subtree)} |")
        return "\n".join(rows)

    # ---- readings -------------------------------------------------------------------------
    def read(self) -> dict[str, np.ndarray]:
        """Graded state: free -> [x y z qw qx qy qz]; slide -> [pos]; hinge -> [angle rad]."""
        q = self.d.qpos
        out = {}
        for it in self.registry:
            if it.kind == "free":
                out[it.name] = q[it.qadr:it.qadr + 7].copy()
            else:
                out[it.name] = q[it.qadr:it.qadr + 1].copy()
        return out

    def finite(self) -> bool:
        return bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all())


@dataclass
class Delta:
    """Per-item change between two readings."""
    name: str
    kind: str
    trans: float        # metres (free: euclidean; slide: |dq|); 0 for hinge
    rot: float          # degrees (free: geodesic; hinge: |dq|); 0 for slide
    status: str         # "changed" | "gray" | "same"


def quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle between unit quaternions: 2*acos(|<q0,q1>|)."""
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def compare(registry: list[Item], r0: dict, r1: dict) -> list[Delta]:
    out = []
    for it in registry:
        a, b = r0[it.name], r1[it.name]
        if it.kind == "free":
            tr = float(np.linalg.norm(b[:3] - a[:3]))
            ro = quat_angle_deg(a[3:7] / np.linalg.norm(a[3:7]), b[3:7] / np.linalg.norm(b[3:7]))
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


def label_from_deltas(deltas: list[Delta]) -> tuple[int, bool]:
    """Returns (y_image, is_gray). y_image is only meaningful when not gray."""
    if any(d.status == "changed" for d in deltas):
        return 1, False
    if any(d.status == "gray" for d in deltas):
        return 0, True
    return 0, False


def label_pair(scene: Scene, r_ref: dict, r_set: dict) -> dict:
    deltas = compare(scene.registry, r_ref, r_set)
    y, gray = label_from_deltas(deltas)
    return {
        "label": y,
        "gray": gray,
        "deltas": deltas,
        "changed_items": [d.name for d in deltas if d.status == "changed"],
        "max_trans": max((d.trans for d in deltas), default=0.0),
        "max_rot": max((d.rot for d in deltas), default=0.0),
    }
