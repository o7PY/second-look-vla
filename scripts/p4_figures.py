"""Part 4 README figures -> docs/figures/p4_arms.png, p4_per_task.png (from results/part4/results_test.json). python scripts/p4_figures.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results" / "part4" / "results_test.json"))
DST = ROOT / "docs" / "figures"
ARMS = ["P0", "PC", "P2r", "SH", "SH-task", "SH-noG"]
COL = {"P0": "#888888", "PC": "#b0b0b0", "P2r": "#d98a2b", "SH": "#2b6fd9", "SH-task": "#6aa0e8", "SH-noG": "#9bbcf0"}


def arms_fig():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (key, title) in zip(axs, [("all", "all 10 tasks (1000 ep.)"), ("seen", "8 seen tasks (800)"), ("held_out", "held-out tasks 6, 8 (200)")]):
        t = R["tables"][key]["arms"]
        y = [100 * t[a]["S"]["est"] for a in ARMS]
        lo = [100 * (t[a]["S"]["est"] - t[a]["S"]["ci"][0]) for a in ARMS]
        hi = [100 * (t[a]["S"]["ci"][1] - t[a]["S"]["est"]) for a in ARMS]
        ax.bar(range(len(ARMS)), y, yerr=[lo, hi], color=[COL[a] for a in ARMS], capsize=3)
        ax.axhline(y[0], color="#555", lw=0.8, ls="--")
        ax.set_xticks(range(len(ARMS)), ARMS, rotation=40, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(50, 82)
    axs[0].set_ylabel("success %  (95% cluster-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(DST / "p4_arms.png", dpi=110)


def task_fig():
    pt = R["per_task"]
    tasks = sorted(pt, key=int)
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, a in enumerate(["P0", "P2r", "SH"]):
        ax.bar(x + (i - 1) * 0.27, [100 * pt[t][a]["S"] for t in tasks], 0.27, label=a, color=COL[a])
    ax.set_xticks(x, [f"task {t}" for t in tasks], rotation=30)
    ax.set_ylabel("success %")
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(DST / "p4_per_task.png", dpi=110)


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    arms_fig(); task_fig()
    for p in sorted(DST.glob("p4_*.png")):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1024:.0f} KB")
