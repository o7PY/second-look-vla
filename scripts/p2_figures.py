"""P2.8 figures (PNG + SVG) -> results/part2/figures/. Reads results/part2/results.json and the TEST logs; computes no new test statistics except the per-task difference CIs.

  python scripts/p2_figures.py

* tradeoff:   success (y) vs silent-failure rate (x), all 11 arms, 95% cluster-bootstrap CIs from results.json (seen tasks, 720 episodes); P0 marked.
* stacked:    success / flagged / silent failure per arm (seen tasks; the three add to 100%).
* sankey:     P0 outcome -> scene truth at the first alarm -> gate verdict -> P3 outcome (counts of episodes, seen tasks; held-out as a second panel).
* per_task:   P3 - P0 and P2 - P0 success difference per task with 95% CI (cluster bootstrap over init states, B = 10,000, 3 noise seeds of an init = one cluster); seen and held-out apart.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "part2" / "figures"
HELD = (6, 8)
ARMS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P3-STAC", "P3-teleport", "P3-seed1", "P3-seed2"]
LABEL = {"P0": "P0 do nothing", "P1": "P1 re-roll", "P2": "P2 blind retry", "P3": "P3 Second Look", "P4": "P4 oracle gate", "P5": "P5 random timing", "P6": "P6 check only (no rewind)",
         "P3-STAC": "P3-STAC trigger", "P3-teleport": "P3-teleport", "P3-seed1": "P3 gate seed 1", "P3-seed2": "P3 gate seed 2"}
COL = {"P0": "#222222", "P1": "#8c8c8c", "P2": "#e08a00", "P3": "#c0392b", "P4": "#7d3c98", "P5": "#2e86c1", "P6": "#16a085", "P3-STAC": "#e67e9a", "P3-teleport": "#d98880", "P3-seed1": "#e6a09a", "P3-seed2": "#e6a09a"}
G_OK, G_FLAG, G_SILENT = "#2e9e55", "#e0a030", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "figure.dpi": 130})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=160 if ext == "png" else None)
    plt.close(fig)
    print("wrote", (OUT / f"{name}.png").relative_to(ROOT), "+ .svg")


def tradeoff(res: dict) -> None:
    arms = res["tables"]["seen"]["arms"]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for nm in ARMS:
        a = arms[nm]
        x, y = 100 * a["SF"]["est"], 100 * a["S"]["est"]
        xe = [[x - 100 * a["SF"]["ci"][0]], [100 * a["SF"]["ci"][1] - x]]
        ye = [[y - 100 * a["S"]["ci"][0]], [100 * a["S"]["ci"][1] - y]]
        main = nm in ("P0", "P2", "P3")
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="*" if nm == "P0" else "o", ms=17 if nm == "P0" else (10 if main else 7), color=COL[nm], ecolor=COL[nm], elinewidth=1.4 if main else 0.9,
                    capsize=2.5, alpha=1.0 if main else 0.8, zorder=5 if main else 3)
    off = {"P0": (14, 10), "P1": (-8, -20), "P2": (10, 8), "P3": (-6, -30), "P4": (22, 26), "P5": (10, -4), "P6": (-60, -34), "P3-STAC": (10, 8), "P3-teleport": (-20, 14), "P3-seed1": (24, -42), "P3-seed2": (-10, 64)}
    for nm in ARMS:
        a = arms[nm]
        main = nm in ("P0", "P2", "P3")
        ax.annotate(LABEL[nm], (100 * a["SF"]["est"], 100 * a["S"]["est"]), textcoords="offset points", xytext=off[nm], fontsize=10 if main else 8.2, color=COL[nm],
                    fontweight="bold" if main else "normal", arrowprops=None if main else dict(arrowstyle="-", color=COL[nm], lw=0.6, alpha=0.7))
    ax.set_xlabel("silent failures (% of episodes: failed, nobody told)   <- better")
    ax.set_ylabel("successes (% of episodes)   better ->")
    ax.set_title("Second Look halves silent failures and costs 3 points of success\n(8 seen LIBERO-Goal tasks, 720 episodes, 95% cluster-bootstrap CIs)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.annotate("", xy=(11, 74.5), xytext=(30, 74.5), arrowprops=dict(arrowstyle="->", color="#555"))
    ax.text(20.5, 75.0, "fewer silent failures", ha="center", fontsize=9, color="#555")
    ax.set_ylim(48, 77)
    save(fig, "tradeoff")


def stacked(res: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    for ax, key, title in ((axs[0], "seen", "8 seen tasks (720 episodes)"), (axs[1], "held_out", "held-out tasks 6, 8 (180 episodes)")):
        arms = res["tables"][key]["arms"]
        y = np.arange(len(ARMS))[::-1]
        S = np.array([100 * arms[a]["S"]["est"] for a in ARMS])
        F = np.array([100 * arms[a]["F"]["est"] for a in ARMS])
        SF = np.array([100 * arms[a]["SF"]["est"] for a in ARMS])
        ax.barh(y, S, color=G_OK, label="success")
        ax.barh(y, F, left=S, color=G_FLAG, label="flagged (stopped, human called)")
        ax.barh(y, SF, left=S + F, color=G_SILENT, label="silent failure")
        for yi, s, f, sf in zip(y, S, F, SF):
            ax.text(s / 2, yi, f"{s:.0f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            if f >= 3:
                ax.text(s + f / 2, yi, f"{f:.0f}", ha="center", va="center", color="black", fontsize=9)
            if sf >= 3:
                ax.text(s + f + sf / 2, yi, f"{sf:.0f}", ha="center", va="center", color="white", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[a] for a in ARMS])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of episodes")
        ax.set_title(title, fontsize=11)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=3, frameon=False)
    fig.suptitle("What happens to the episodes: success / flagged / silent failure", fontsize=12.5)
    save(fig, "stacked")


def load_eps(d: Path) -> list[dict]:
    eps = []
    for f in sorted(d.glob("ep_*.json.gz")):
        r = json.load(gzip.open(f))
        p0 = r["arms"]["P0"]
        arms = {nm: (r["arms"].get(nm) if r["arms"].get(nm) is not None else dict(p0, events=[], flagged=False)) for nm in ARMS}
        eps.append({"task": r["task"], "init": r["init"], "seed": r["seed_id"], "arms": arms})
    return eps


def flows(eps: list[dict]) -> Counter:
    """path counts: (P0 outcome, truth at the first alarm, gate verdict, P3 outcome)."""
    c = Counter()
    for e in eps:
        a, p0 = e["arms"]["P3"], e["arms"]["P0"]
        g = [x for x in a["events"] if x["kind"] == "gate"]
        o0 = "P0 success" if p0["success"] else "P0 failure"
        out = "success" if a["success"] else ("flagged" if a["flagged"] else "silent failure")
        if not g:
            c[(o0, "no alarm", "no check", out)] += 1
            continue
        t = "gray" if g[0]["truth_gray"] else ("changed" if g[0]["truth"] else "unchanged")
        c[(o0, t, "gate CHANGED" if g[0]["verdict"] == "CHANGED" else "gate UNCHANGED", out)] += 1
    return c


def alluvial(ax, paths: Counter, stages: list[list[str]], colors: dict, title: str, label_min: int = 0) -> None:
    n_total = sum(paths.values())
    gap = 0.035 * n_total
    pos = []                                                  # per stage: node -> (y0, size)
    for si, nodes in enumerate(stages):
        sizes = {nd: sum(v for p, v in paths.items() if p[si] == nd) for nd in nodes}
        nodes_present = [nd for nd in nodes if sizes[nd] > 0]
        y, d = 0.0, {}
        for nd in nodes_present:
            d[nd] = (y, sizes[nd])
            y += sizes[nd] + gap
        pos.append(d)
    xs = np.arange(len(stages)) * 3.4
    bw = 0.55
    cursor_out = [{nd: v[0] for nd, v in d.items()} for d in pos]            # next free y at the right edge of each node
    cursor_in = [{nd: v[0] for nd, v in d.items()} for d in pos]
    order = sorted(paths.items(), key=lambda kv: (stages[0].index(kv[0][0]), stages[1].index(kv[0][1]), stages[2].index(kv[0][2]), stages[3].index(kv[0][3])))
    for si in range(len(stages) - 1):
        agg = Counter()
        for p, v in paths.items():
            agg[(p[si], p[si + 1], p[3] if si == 2 else (p[0] if si == 0 else p[1]))] += v
        # one ribbon per (node_i, node_j, colour key); draw in node order so the ribbons do not cross more than needed
        items = sorted(agg.items(), key=lambda kv: (stages[si].index(kv[0][0]), stages[si + 1].index(kv[0][1]), stages[3].index(kv[0][2]) if si == 2 else 0))
        for (a, b, _), v in items:
            y0, y1 = cursor_out[si][a], cursor_in[si + 1][b]
            cursor_out[si][a] += v
            cursor_in[si + 1][b] += v
            x0, x1 = xs[si] + bw, xs[si + 1] - bw
            xm = (x0 + x1) / 2
            verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + v), (xm, y1 + v), (xm, y0 + v), (x0, y0 + v), (x0, y0)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colors.get(b if si == 2 else a, "#bbbbbb"), edgecolor="none", alpha=0.45))
    for si, d in enumerate(pos):
        for nd, (y0, sz) in d.items():
            ax.add_patch(plt.Rectangle((xs[si] - bw, y0), 2 * bw, sz, facecolor=colors.get(nd, "#777777"), edgecolor="white", lw=0.8))
            txt = f"{nd}  {sz}"
            if si == 0:
                ax.text(xs[si] - bw - 0.08, y0 + sz / 2, txt, ha="right", va="center", fontsize=9.5, fontweight="bold")
            elif si == len(stages) - 1:
                ax.text(xs[si] + bw + 0.08, y0 + sz / 2, txt, ha="left", va="center", fontsize=9.5, fontweight="bold")
            else:
                big = sz > 0.045 * n_total
                ax.text(xs[si], y0 + sz / 2, f"{nd.replace('gate ', '')}\n{sz}" if big else f"{nd.replace('gate ', '')} {sz}", ha="center", va="center", fontsize=8.8 if big else 7.2,
                        color="white" if big else "black", fontweight="bold")
    for si, name in enumerate(["what P0 did", "scene at 1st alarm\n(simulator truth)", "gate verdict", "what P3 did"]):
        ax.text(xs[si], -0.06 * n_total, name, ha="center", va="top", fontsize=9.5, color="#333")
    ax.set_xlim(xs[0] - 2.6, xs[-1] + 2.6)
    ax.set_ylim(-0.12 * n_total, max(sum(v for _, v in d.values()) + gap * len(d) for d in pos) + 0.02 * n_total)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def sankey(eps: list[dict]) -> None:
    stages = [["P0 success", "P0 failure"], ["no alarm", "changed", "unchanged", "gray"], ["no check", "gate CHANGED", "gate UNCHANGED"], ["success", "flagged", "silent failure"]]
    colors = {"P0 success": G_OK, "P0 failure": "#7f8c8d", "no alarm": "#95a5a6", "changed": "#2874a6", "unchanged": "#a9884a", "gray": "#bdc3c7", "no check": "#95a5a6",
              "gate CHANGED": "#2874a6", "gate UNCHANGED": "#a9884a", "success": G_OK, "flagged": G_FLAG, "silent failure": G_SILENT}
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1.0, 1.0]})
    seen = [e for e in eps if e["task"] not in HELD]
    held = [e for e in eps if e["task"] in HELD]
    alluvial(axs[0], flows(seen), stages, colors, f"8 seen tasks ({len(seen)} episodes)")
    alluvial(axs[1], flows(held), stages, colors, f"held-out tasks 6, 8 ({len(held)} episodes)")
    fig.suptitle("P3 Second Look, first alarm of every episode: what the scene did, what the gate said, what happened", fontsize=13)
    save(fig, "first_alarm_flow")


def per_task(eps: list[dict], B: int = 10000) -> None:
    rng = np.random.default_rng(0)
    rows = {}
    for t in range(10):
        es = [e for e in eps if e["task"] == t]
        inits = sorted({e["init"] for e in es})
        d = {}
        for arm in ("P3", "P2"):
            v = np.array([np.mean([float(e["arms"][arm]["success"]) - float(e["arms"]["P0"]["success"]) for e in es if e["init"] == i]) for i in inits])
            idx = rng.integers(0, len(v), (B, len(v)))
            bs = v[idx].mean(1)
            d[arm] = (100 * v.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5))
        rows[t] = d
    seen = [t for t in range(10) if t not in HELD]
    order = seen + list(HELD)
    ypos = {}
    y = 0.0
    for t in order:
        if t == HELD[0]:
            y += 0.9
        ypos[t] = y
        y += 1.0
    fig, ax = plt.subplots(figsize=(9, 7.3))
    for arm, off, col in (("P3", -0.17, COL["P3"]), ("P2", 0.17, COL["P2"])):
        for t in order:
            m, lo, hi = rows[t][arm]
            ax.errorbar(m, ypos[t] + off, xerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=6, capsize=2.5, elinewidth=1.3, label=LABEL[arm] + " - P0" if t == order[0] else None)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks([ypos[t] for t in order])
    names = {0: "open the middle drawer", 1: "bowl on the stove", 2: "wine bottle on the cabinet", 3: "open top drawer, bowl inside", 4: "bowl on top of the cabinet", 5: "push plate to the stove front",
             6: "cream cheese in the bowl", 7: "turn on the stove", 8: "bowl on the plate", 9: "wine bottle on the rack"}
    ax.set_yticklabels([f"task {t}{' (held-out)' if t in HELD else ''}: {names[t]}" for t in order], fontsize=9)
    ax.invert_yaxis()
    ax.axhspan(ypos[HELD[0]] - 0.75, ypos[HELD[1]] + 0.6, color="#f3e9c9", alpha=0.5, zorder=0)
    ax.set_xlabel("change in success rate vs P0 (percentage points, mean over 30 init states x 3 seeds)")
    ax.set_title("Per task: P3 Second Look and P2 blind retry minus P0\n(95% cluster-bootstrap CIs over init states; held-out tasks shaded, exploratory)", fontsize=11.5)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "per_task")
    json.dump({str(t): {a: list(map(float, v)) for a, v in d.items()} for t, d in rows.items()}, open(OUT / "per_task.json", "w"), indent=1)


def main() -> None:
    res = json.load(open(ROOT / "results" / "part2" / "results.json"))
    tradeoff(res)
    stacked(res)
    eps = load_eps(ROOT / "results" / "part2" / "test")
    sankey(eps)
    per_task(eps)


if __name__ == "__main__":
    main()
