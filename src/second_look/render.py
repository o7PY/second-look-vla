"""Deterministic rendering (SPEC §3.5): agentview RGB + segmentation, 2x supersampled.

Convention (written once, used everywhere): the raw OpenGL image from MuJoCo is rotated 180°
(``img[::-1, ::-1]``), exactly what LeRobot does for LIBERO agentview. Every RGB image and every mask
in the dataset is stored in this upright orientation.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

from second_look.scene import CAMERA, Scene, is_robot_name

RENDER_PX = 448          # supersampled render size
OUT_PX = 224             # stored / model size
OFFSAMPLES = 0           # MSAA off: GPU MSAA is not repeatable on this machine

cv2.setNumThreads(1)     # INTER_AREA is deterministic anyway; belt and braces


def patch() -> None:
    """MSAA off for every robosuite render context (idempotent; must run before the context is built)."""
    from robosuite.utils import binding_utils as bu
    cls = bu.MjRenderContext
    if getattr(cls, "_sl_patched", False):
        return
    orig = cls._set_mujoco_context_and_buffers

    def _set(self):
        self.model._model.vis.quality.offsamples = OFFSAMPLES
        orig(self)

    cls._set_mujoco_context_and_buffers = _set
    cls._sl_patched = True


def upright(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def activate(scene: Scene) -> None:
    """Make this scene's GL context current. robosuite renders into whatever context is current; after another Scene was created or closed (closing
    destroys its context) that is not this scene's, and the render comes out as noise (found 2026-10-06, tests/test_b2_render_state.py)."""
    ctx = getattr(scene.sim, "_render_context_offscreen", None)
    gl = getattr(ctx, "gl_ctx", None)
    if gl is not None and hasattr(gl, "make_current"):
        gl.make_current()


def render_rgb(scene: Scene) -> np.ndarray:
    """(224,224,3) uint8: 448 render, MSAA off, cv2.INTER_AREA down-sample."""
    activate(scene)
    raw = scene.sim.render(width=RENDER_PX, height=RENDER_PX, camera_name=CAMERA)
    return cv2.resize(upright(raw), (OUT_PX, OUT_PX), interpolation=cv2.INTER_AREA)


def render_geom_ids(scene: Scene) -> np.ndarray:
    """(448,448) int32 geom id per pixel (-1 = background/sky), upright, full resolution."""
    activate(scene)
    sim = scene.sim
    ctx = sim._render_context_offscreen
    cam_id = sim.model.camera_name2id(CAMERA)
    ctx.render(width=RENDER_PX, height=RENDER_PX, camera_id=cam_id, segmentation=True)
    rgb = np.empty((RENDER_PX, RENDER_PX, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, RENDER_PX, RENDER_PX), con=ctx.con)
    code = rgb[..., 0].astype(np.int32) + (rgb[..., 1].astype(np.int32) << 8) + (rgb[..., 2].astype(np.int32) << 16)
    scn = ctx.scn
    lut = np.full(scn.ngeom + 2, -1, dtype=np.int32)       # code -> geom id
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        if g.segid != -1 and g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
            lut[g.segid + 1] = g.objid
    code[code >= scn.ngeom + 2] = 0
    return upright(lut[code])


class Masks:
    """Geom-id image -> robot mask and per-item index image, at 224 (nearest-neighbour)."""

    def __init__(self, scene: Scene):
        m = scene.m
        gb = m.geom_bodyid
        self.geom_robot = np.array([is_robot_name(scene.body_names[b]) for b in gb], dtype=bool)
        # geom -> registry item index (-1 = none). Later items win; subtrees do not overlap in LIBERO.
        self.geom_item = np.full(m.ngeom, -1, dtype=np.int16)
        body_item = np.full(m.nbody, -1, dtype=np.int16)
        for k, it in enumerate(scene.registry):
            body_item[it.subtree] = k
        self.geom_item = body_item[gb].astype(np.int16)

    def __call__(self, geom_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = geom_ids >= 0
        safe = np.where(valid, geom_ids, 0)
        robot = valid & self.geom_robot[safe]
        item = np.where(valid, self.geom_item[safe], -1).astype(np.int16)
        robot = cv2.resize(robot.astype(np.uint8), (OUT_PX, OUT_PX), interpolation=cv2.INTER_NEAREST).astype(bool)
        item = cv2.resize(item, (OUT_PX, OUT_PX), interpolation=cv2.INTER_NEAREST)
        return robot, item


def render_frame(scene: Scene, masks: Masks) -> dict:
    """RGB + robot mask + item-index image for the scene's current state."""
    rgb = render_rgb(scene)
    robot, item = masks(render_geom_ids(scene))
    return {"rgb": rgb, "robot": robot, "item": item}


def to_patch_mask(mask: np.ndarray, frac: float = 0.10, grid: int = 16) -> np.ndarray:
    """224x224 bool -> 16x16 bool: patch is set if >= ``frac`` of its pixels are set."""
    p = mask.shape[0] // grid
    cov = mask.reshape(grid, p, grid, p).mean(axis=(1, 3))
    return cov >= frac


def dilate_patches(pm: np.ndarray, n: int = 1) -> np.ndarray:
    k = np.ones((2 * n + 1, 2 * n + 1), np.uint8)
    return cv2.dilate(pm.astype(np.uint8), k).astype(bool)


def arm_patches(robot_ref: np.ndarray, robot_set: np.ndarray) -> np.ndarray:
    """§3.4: arm if >= 10% robot pixels in EITHER frame, dilated by 1 patch."""
    return dilate_patches(to_patch_mask(robot_ref) | to_patch_mask(robot_set), 1)


def changed_patches(chg_ref: np.ndarray, chg_set: np.ndarray) -> np.ndarray:
    """§3.4: changed if >= 10% changed-object pixels in EITHER frame."""
    return to_patch_mask(chg_ref) | to_patch_mask(chg_set)
