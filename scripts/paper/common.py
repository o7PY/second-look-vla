"""Shared helpers of the paper-asset scripts: source loading, key-path resolution, plotting style. Everything reads result files; nothing is typed by hand."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

# Okabe-Ito colour-blind-safe palette
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "pink": "#CC79A7", "grey": "#7F7F7F"}


@lru_cache(maxsize=None)
def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def resolve(rel: str, path):
    """Walk a key path (tuple of tokens) through a JSON file; list tokens are ints, dict tokens are strings."""
    cur = load(rel)
    for t in path:
        cur = cur[int(t)] if isinstance(cur, list) else cur[str(t)]
    return cur


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "svg.hashsalt": "second-look",
    })
    return plt


def save(fig, name: str):
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / "figures" / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    fig.savefig(PAPER / "figures" / f"{name}.png")
