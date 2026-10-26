"""Camera-image sanity check (added 2026-10-10 after the render-context bug of src/second_look/glfix.py). Called on the agentview image of EVERY decision (second_look.loop.decide) and on every
frame DeltaDINO is shown (second_look.gate.Frames.render_flat); a failing image raises `DegenerateImage` and crashes the episode loudly instead of letting the policy act on garbage.

The corrupted renders of the bug were striped tiles: strong horizontal-neighbour contrast with little vertical contrast, and an image that repeats itself within a third of its
size. Thresholds were set on the saved corrupted vs good images (tests/p2/fixtures/camera_images.npz) and on 20 stored good renders of every kind:
  agentview: std >= 12  (blank / constant image)           good 46-70
             anisotropy  dx/dy and dy/dx < 2.5             good 0.8-1.2, corrupted 3.7 (and 5.5)
             tiling      min mean-abs difference over shifts 16..h/2 (either axis) >= 10    good 16-30, corrupted 4.9-6.2
  wrist:     anisotropy rule only (a wrist view onto a plain surface can legitimately have low contrast)
"""
from __future__ import annotations

import numpy as np

STD_MIN = 12.0
ANISO_MAX = 2.5
TILE_MIN = 10.0


class DegenerateImage(RuntimeError):
    pass


def _gray(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, np.float32).mean(-1)


def measures(img: np.ndarray) -> dict:
    g = _gray(img)
    dx = float(np.abs(np.diff(g, axis=1)).mean())
    dy = float(np.abs(np.diff(g, axis=0)).mean())
    h = g.shape[0]
    shifts = range(16, h // 2 + 1, 4)
    per_y = min(float(np.abs(g[s:] - g[:-s]).mean()) for s in shifts)
    per_x = min(float(np.abs(g[:, s:] - g[:, :-s]).mean()) for s in shifts)
    return {"std": float(np.asarray(img, np.float32).std()), "dx": dx, "dy": dy, "aniso": max(dx, dy) / (min(dx, dy) + 1e-6), "per_x": per_x, "per_y": per_y}


def problems(img: np.ndarray, *, agentview: bool = True) -> list[str]:
    m = measures(img)
    out = []
    if max(m["dx"], m["dy"]) > 8.0 and m["aniso"] > ANISO_MAX:
        out.append(f"striped (dx {m['dx']:.1f}, dy {m['dy']:.1f}, ratio {m['aniso']:.1f} > {ANISO_MAX})")
    if agentview:
        if m["std"] < STD_MIN:
            out.append(f"blank (std {m['std']:.1f} < {STD_MIN})")
        if min(m["per_x"], m["per_y"]) < TILE_MIN:
            out.append(f"repeats itself (min shift difference {min(m['per_x'], m['per_y']):.1f} < {TILE_MIN})")
    return out


def check_image(img: np.ndarray, where: str, *, agentview: bool = True) -> None:
    p = problems(img, agentview=agentview)
    if p:
        raise DegenerateImage(f"degenerate camera image at {where}: " + "; ".join(p))


def check_obs(obs: dict, where: str) -> None:
    """lerobot's LiberoEnv observation: pixels {image: agentview, image2: wrist}."""
    check_image(obs["pixels"]["image"], where + " agentview", agentview=True)
    if "image2" in obs["pixels"]:
        check_image(obs["pixels"]["image2"], where + " wrist", agentview=False)
