"""Part 4 shared loaders: alarm cases from GEN logs, per-episode outcome tables for DEV / TEST (reads logs only)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELD = (6, 8)
R_ARMS = ["R1", "R2", "R3", "R4"]
C_ARMS = ["C1", "C2", "C3", "C4"]


def load(d: str) -> dict:
    out = {}
    for f in sorted((ROOT / d).glob("ep_*.json.gz")):
        r = json.load(gzip.open(f))
        out[(r["task"], r["init"], r["seed_id"])] = r
    return out


def sh_event(arm: dict):
    for e in arm.get("events", []):
        if e["kind"] == "sh":
            return e
    return None


def gate_event(arm: dict):
    for e in arm.get("events", []):
        if e["kind"] == "gate":
            return e
    return None


def gen_cases(recs: dict) -> tuple[list[dict], dict]:
    """One case per GEN episode whose first alarm reached the decision (features exist). Excluded (counted): no alarm; the pause itself finished or exhausted the episode (no decision to make)."""
    cases, skipped = [], {"no_alarm": 0, "ended_in_pause": 0}
    for k, r in recs.items():
        arms = r["arms"]
        if "R1" not in arms:
            skipped["no_alarm"] += 1
            continue
        ev = sh_event(arms["R1"])
        if ev is None:
            skipped["ended_in_pause"] += 1
            continue
        xs = [sh_event(arms[a])["x"] for a in R_ARMS + C_ARMS]
        assert all(x == xs[0] for x in xs), f"features differ between branches of {k}"
        g = gate_event(arms["R1"])
        cases.append({"key": k, "task": k[0], "init": k[1], "seed": k[2], "x": np.array(xs[0], float), "y_r": np.array([bool(arms[a]["success"]) for a in R_ARMS], int),
                      "y_c": np.array([bool(arms[a]["success"]) for a in C_ARMS], int), "p0": bool(arms["P0"]["success"]), "truth": None if g is None else ("gray" if g["truth_gray"] else ("changed" if g["truth"] else "unchanged")),
                      "d": ev["d"], "elapsed": ev["elapsed"]})
    return cases, skipped


def arm_of(r: dict, name: str) -> dict:
    """An arm absent from the log never alarmed: it equals P0."""
    a = r["arms"].get(name)
    return a if a is not None else dict(r["arms"]["P0"], events=[], rewinds=0, invalid_rewinds=0, cut_short=False, flagged=False, flag_reason=None)
