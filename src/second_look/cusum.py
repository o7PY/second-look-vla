"""One-sided CUSUM alarm with a split-conformal height (SPEC_part2.md §6, P2.2/P2.3).

    S_t = max(0, S_{t-1} + x_t - mu0 - k),   k = 0.25 sigma,   alarm at the first S_t > h.

mu0 and sigma are the mean and std of the detector score pooled over all decisions of the CAL *successes*; h is the ceil((M+1)(1-alpha))-th smallest
per-episode CUSUM peak over those M successes (split-conformal: a new success episode peaks above h with probability <= alpha). The state is two floats,
so a rewind restores it by copying (`CUSUM.state` / `CUSUM.restore`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_FACTOR = 0.25


@dataclass(frozen=True)
class CusumParams:
    mu0: float
    sigma: float
    h: float
    k: float

    @classmethod
    def make(cls, mu0: float, sigma: float, h: float) -> "CusumParams":
        return cls(float(mu0), float(sigma), float(h), K_FACTOR * float(sigma))


class CUSUM:
    def __init__(self, p: CusumParams):
        self.p, self.S = p, 0.0

    def update(self, x: float) -> tuple[float, bool]:
        """Feed one decision's score. Returns (S_t, alarm)."""
        self.S = max(0.0, self.S + float(x) - self.p.mu0 - self.p.k)
        return self.S, self.S > self.p.h

    @property
    def state(self) -> float:
        return self.S

    def restore(self, s: float) -> None:
        self.S = float(s)


def peaks(episodes: list[np.ndarray], mu0: float, k: float) -> np.ndarray:
    """Per-episode maximum of S_t (S_0 = 0) for score sequences `episodes`."""
    out = []
    for x in episodes:
        s, m = 0.0, 0.0
        for v in x:
            s = max(0.0, s + float(v) - mu0 - k)
            m = max(m, s)
        out.append(m)
    return np.asarray(out)


def conformal_height(pk: np.ndarray, alpha: float) -> float:
    """The ceil((M+1)(1-alpha))-th smallest peak (1-indexed); +inf if that rank exceeds M (too few calibration episodes)."""
    m = len(pk)
    r = math.ceil((m + 1) * (1 - alpha))
    return float("inf") if r > m else float(np.sort(pk)[r - 1])


def calibrate(success_scores: list[np.ndarray], alpha: float = 0.1) -> CusumParams:
    """Parameters from the score sequences (one per decision step) of the CAL success episodes."""
    pooled = np.concatenate([np.asarray(x, np.float64) for x in success_scores])
    mu0, sigma = float(pooled.mean()), float(pooled.std())
    k = K_FACTOR * sigma
    return CusumParams(mu0, sigma, conformal_height(peaks(success_scores, mu0, k), alpha), k)


def first_alarm(x: np.ndarray, p: CusumParams) -> int | None:
    """Index of the first decision whose S_t > h, else None (offline, for CAL statistics and tests)."""
    c = CUSUM(p)
    for i, v in enumerate(x):
        if c.update(v)[1]:
            return i
    return None
