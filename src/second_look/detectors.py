"""Failure-detector scores for the closed loop (SPEC_part2.md §6, P2.2): accel (primary) and STAC (secondary). Independent numpy ports of the formulas; the
reference implementation (third_party/fm-geometry, fmaccel) is used only by test G6, on identical inputs.

accel (Rao et al., Algorithm 1) on the executed window: the 10 executed actions x 7 dims of every Euler iterate (11 points for 10 steps), z-scored per
coordinate by a fixed scale (std of the flattened window over decisions and iterates, from CAL), then for the prefix of p = 9 iterates:

    v_k = x_{k+1} - x_k  (k = 0 .. p-2),   accel_p = p * sum_{k=1}^{p-1} ||v_k - v_{k-1}|| / sum_{k=0}^{p-1} ||v_k||

with p counted in denoise steps as fmaccel's `accel_prefix:7` (prefix of the first 10 points = 9 steps); the sums are over the prefix's steps and jerks.

STAC (Sentinel): biased RBF-MMD^2 between the previous decision's K resamples (steps `n_exec:` = the continuation past the executed window, 40 steps) and the
current decision's K resamples (steps `:chunk-n_exec`, the same 40 absolute future steps), all 7 dims, kernel exp(-gamma ||.||^2) with
gamma = 1 / (2 * median of the pooled SQUARED pairwise distances > 0). First decision: 0. Fewer than 2 samples: 0.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
N_EXEC = 10
PREFIX_STEPS = 9                       # fmaccel accel_prefix:7  ->  first 10 iterates = 9 denoise steps


# ------------------------------------------------------------------------------------------------ accel
def exec_window(iterates: np.ndarray, n_exec: int = N_EXEC) -> np.ndarray:
    """(T+1, chunk, 7) -> (T+1, n_exec * 7) flattened executed window."""
    return np.asarray(iterates, np.float32)[:, :n_exec, :].reshape(iterates.shape[0], -1)


def fit_accel_scale(windows: list[np.ndarray]) -> np.ndarray:
    """Per-coordinate scale, once on CAL: std over (decisions, iterates) of the flattened executed window. `windows`: list of (T+1, n_exec*7)."""
    flat = np.concatenate([np.asarray(w, np.float32) for w in windows], axis=0)
    return flat.std(axis=0).astype(np.float32)


def accel(iterates: np.ndarray, scale: np.ndarray, *, n_exec: int = N_EXEC, steps: int = PREFIX_STEPS) -> float:
    """accel_p of one decision. `iterates` (T+1, chunk, 7); `scale` (n_exec*7,)."""
    xz = exec_window(iterates, n_exec) / (np.asarray(scale, np.float32) + EPS)       # (T+1, KA)
    xz = xz[: steps + 1]                                                              # first steps+1 points = `steps` denoise steps
    v = np.diff(xz, axis=0)                                                           # (steps, KA)
    vlen = np.linalg.norm(v, axis=1)
    jerk = np.linalg.norm(np.diff(v, axis=0), axis=1)                                 # (steps-1,)
    return float(steps * jerk.sum() / (vlen.sum() + EPS))


# ------------------------------------------------------------------------------------------------ STAC
def _sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def mmd2_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0
    kxx, kyy, kxy = _sq(X, X), _sq(Y, Y), _sq(X, Y)
    pooled = np.concatenate([kxx[np.triu_indices(n, 1)], kyy[np.triu_indices(m, 1)], kxy.ravel()])
    pos = pooled[pooled > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    g = 1.0 / (2.0 * med + 1e-12)
    return float(max(np.exp(-g * kxx).mean() + np.exp(-g * kyy).mean() - 2.0 * np.exp(-g * kxy).mean(), 0.0))


def stac(cur: np.ndarray | None, prev: np.ndarray | None, *, n_exec: int = N_EXEC) -> float:
    """`cur`, `prev`: (K, chunk, 7) resamples of this and the previous decision (None -> 0.0)."""
    if cur is None or prev is None or cur.shape[0] < 2 or prev.shape[0] < 2:
        return 0.0
    overlap = cur.shape[1] - n_exec
    X = np.asarray(cur[:, :overlap, :], np.float32).reshape(cur.shape[0], -1)
    Y = np.asarray(prev[:, n_exec:, :], np.float32).reshape(prev.shape[0], -1)
    return mmd2_rbf(X, Y)
