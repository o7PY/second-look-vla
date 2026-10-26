"""Determinism flags and the Part 2 noise keys (SPEC_part2.md §2.4).

SmolVLA's flow-matching noise for one decision is a pure function of WHERE it is used:

    sha256(split | task | init | seed_id | intervention_idx | decision | k)

`intervention_idx` counts the rewinds made so far in the episode (0 before the first); it is the string "reroll" for the P1 arm after its alarm.
The arm name is deliberately NOT part of the key, so every arm is bit-identical to the baseline until its first intervention, and P2 / P3 are
bit-identical until the first time the gate says "changed". `k = 0` is the executed chunk, `k = 1..8` are the STAC resamples.
Ported from the old project's d2r/seeding.py (src/second_look/_legacy_d2r/seeding.py); the key fields are the Part 2 ones.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

REROLL = "reroll"
P5NC1B = "p5nc1b"                # Part 5 (P5.1b): later decisions of a retry whose FIRST decision keeps a P5.0 noise: "p5nc1b_<j>_<m>" (draw j, stream m)
P5NC = "p5nc"                    # Part 5 (P5.0): noise-ceiling streams, intervention index "p5nc<j>" (j = 0, 1, ...); no Part 1-4 key uses it


def set_determinism() -> None:
    """Settings validated as bitwise-reproducible on this machine in the old project (Phase 0 T1/T5). Call AFTER anything that flips them back."""
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _digest(*fields) -> bytes:
    return hashlib.sha256("|".join(str(f) for f in fields).encode()).digest()


def _check_intervention(intervention_idx) -> None:
    if not (intervention_idx == REROLL or (isinstance(intervention_idx, str) and intervention_idx.startswith(P5NC) and intervention_idx[len(P5NC):].isdigit()) or (isinstance(intervention_idx, str) and re.fullmatch(P5NC1B + r"_\d+_\d+", intervention_idx)) or (isinstance(intervention_idx, (int, np.integer)) and not isinstance(intervention_idx, bool) and intervention_idx >= 0)):
        raise ValueError(f"intervention_idx must be an int >= 0, {REROLL!r} 'p5nc<j>' or 'p5nc1b_<j>_<m>', got {intervention_idx!r}")


def noise_seed(split: str, task: int, init: int, seed_id: str, intervention_idx, decision: int, k: int = 0) -> int:
    """Stable 63-bit seed for the noise of decision `decision`, sample `k`. No arm name, by design."""
    _check_intervention(intervention_idx)
    if decision < 0 or k < 0:
        raise ValueError("decision and k must be >= 0")
    d = _digest(split, int(task), int(init), seed_id, intervention_idx, int(decision), int(k))
    return int.from_bytes(d[:8], "little") & ((1 << 63) - 1)


def fm_noise(seed: int, shape: tuple[int, ...], device="cuda"):
    """Standard-normal fp32 noise from a dedicated CPU generator, then moved: the values do not depend on the GPU's RNG."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def decision_noise(split: str, task: int, init: int, seed_id: str, intervention_idx, decision: int, ks=(0,), *,
                   chunk_size: int = 50, max_action_dim: int = 32, device="cuda"):
    """Noise for samples `ks` of one decision stacked on the batch axis -> (len(ks), chunk, D)."""
    import torch
    return torch.cat([fm_noise(noise_seed(split, task, init, seed_id, intervention_idx, decision, k), (1, chunk_size, max_action_dim), device) for k in ks])


def p5_uniform(split: str, task: int, init: int, seed_id: str, intervention_idx, decision: int) -> float:
    """The uniform [0, 1) draw P5 compares with its alarm hazard: sha256("p5", split, task, init, seed_id, intervention_idx, decision)."""
    _check_intervention(intervention_idx)
    d = _digest("p5", split, int(task), int(init), seed_id, intervention_idx, int(decision))
    return int.from_bytes(d[:8], "little") / float(1 << 64)
