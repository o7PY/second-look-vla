"""paper/figures/F1..F6 (PDF + PNG), colour-blind-safe (Okabe-Ito), no dual axes, from paper/numbers.json and the Part 5 result files.
  python scripts/paper/build_figures.py [F1 F2 ...]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import C, PAPER, load, save, style  # noqa: E402

plt = style()
LED = {r["id"]: r for r in json.loads((PAPER / "numbers.json").read_text())["rows"]}
V = lambda i: 100 * LED[i]["value"]  # noqa: E731
CI = lambda i: [100 * x for x in LED[i]["ci"]]  # noqa: E731


def f1():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=C["black"], lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8)

    def arrow(a, b, text="", dy=0.0, dx=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=C["black"]))
        if text:
            ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + 1.6 + dy, text, ha="center", va="bottom", fontsize=6.8, style="italic")

    box(1, 32, 17, 10, "SmolVLA runs\n(10-action chunks)", "#DDDDDD")
    box(22, 32, 17, 10, "Alarm: CUSUM on\nplan acceleration", C["sky"])
    box(43, 32, 17, 10, "Pause 10 steps,\nscene settles", "#DDDDDD")
    box(64, 32, 17, 10, "DeltaDINO: has\nthe scene changed?", C["orange"])
    box(86, 33, 13, 8, "Flag the\nepisode", C["pink"])
    box(64, 12, 17, 10, "Arm-only rewind\nto the checkpoint", "#DDDDDD")
    box(43, 12, 17, 10, "Retry with fresh\nflow noise", C["green"])
    box(22, 12, 17, 10, "Continue; no\nfurther retry", "#DDDDDD")
    box(1, 12, 17, 10, "Success or\ntimeout", "#DDDDDD")
    arrow((19, 37), (21.2, 37)); arrow((40, 37), (42.2, 37)); arrow((61, 37), (63.2, 37))
    arrow((82, 37), (85.2, 37)); ax.text(83.6, 38.2, "yes", ha="center", va="bottom", fontsize=6.5, style="italic")
    arrow((72.5, 31.4), (72.5, 22.6)); ax.text(74, 26.5, "no", ha="left", va="center", fontsize=6.8, style="italic")
    arrow((63.2, 17), (61, 17)); arrow((42.2, 17), (40, 17)); arrow((21.2, 17), (19, 17))
    ax.text(50, 3.2, "Parts 2-4 vary the rule at each step: P2 retries without the check, P3 is the loop above, PA gives a grace on the first 'changed',\nPart 4 chooses retry or continue per alarm with a value model; Part 5 asks which fresh noise to retry with.", ha="center", fontsize=6.8)
    save(fig, "F1_pipeline"); plt.close(fig)


def f2():
    panels = [("Part 2 (seen tasks, 720 episodes)", "p2.seen", [("P0", "P0"), ("P2", "P2 blind retry"), ("P3", "P3"), ("P5", "P5 random timing"), ("P6", "P6 check only")], C["blue"]),
              ("Part 3 (TEST, 1000 episodes)", "p3.all", [("P0", "P0"), ("P3", "P3"), ("PA", "PA"), ("PAc", "PAc")], C["verm"]),
              ("Part 4 (TEST, 1000 episodes)", "p4", [("P0", "P0"), ("PC", "PC"), ("P2r", "P2r"), ("SH", "SH"), ("SH-task", "SH-task"), ("SH-noG", "SH-noG"), ("SH+F", "SH+F")], C["green"])]
    offs = {"P0": (8, 6), "P2": (8, 4), "P3": (8, -2), "P5": (8, 2), "P6": (8, -9), "PA": (8, 2), "PAc": (8, -2)}
    offs4 = {"P0": (-30, 22), "PC": (-42, 10), "P2r": (-18, -26), "SH": (-14, 26), "SH-task": (20, 20), "SH-noG": (34, -4), "SH+F": (10, 6)}
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.38, wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for k, (ax, (title, pre, arms, col)) in enumerate(zip(axes, panels)):
        for arm, lab in arms:
            i = f"{pre}.{arm}"
            x, y = V(i + ".S"), V(i + ".SF")
            ax.errorbar(x, y, xerr=[[x - CI(i + ".S")[0]], [CI(i + ".S")[1] - x]], yerr=[[y - CI(i + ".SF")[0]], [CI(i + ".SF")[1] - y]], fmt="o" if arm != "P0" else "s", ms=5,
                        color=col if arm != "P0" else C["black"], ecolor="#BBBBBB", elinewidth=0.6, capsize=0, zorder=2)
            if k == 2:
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=offs4[arm], fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.5, color=C["grey"]), ha="center", va="center")
            else:
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=offs[arm], fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("success %"); ax.set_ylabel("silent failures % (lower is better)")
        ax.margins(0.12)
    axes[2].set_xlim(63.5, 76)
    save(fig, "F2_success_vs_silent"); plt.close(fig)


def f3():
    spec = [("Part 2 (seen, 720)", "p2.seen", ["P1", "P2", "P3", "P5", "P6"], 720), ("Part 3 (TEST, 1000)", "p3.all", ["P3", "PA", "PAc"], 1000),
            ("Part 4 (TEST, 1000)", "p4", ["PC", "P2r", "SH", "SH-task", "SH-noG", "SH+F"], 1000)]
    labels, res, har, grp = [], [], [], []
    for g, (title, pre, arms, n) in enumerate(spec):
        for a in arms:
            labels.append(f"{a}"); grp.append(g)
            res.append(100 * LED[f"{pre}.{a}.rescue"]["value"] / n); har.append(100 * LED[f"{pre}.{a}.harm"]["value"] / n)
    y = []
    pos = 1.0
    for k, g in enumerate(grp):
        if k and g != grp[k - 1]:
            pos += 1.2
        y.append(pos); pos += 1
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.barh(y, res, color=C["green"], height=0.7, label="rescues (P0 failed, arm succeeded)")
    ax.barh(y, [-h for h in har], color=C["verm"], height=0.7, label="harms (P0 succeeded, arm failed)")
    ax.plot([r - h for r, h in zip(res, har)], y, "D", color=C["black"], ms=3.5, label="net")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.axvline(0, color=C["black"], lw=0.7)
    for g, (title, *_r) in enumerate(spec):
        ys = [yy for yy, gg in zip(y, grp) if gg == g]
        ax.text(-15.8, min(ys) - 0.95, title, va="center", ha="left", fontsize=7.5, fontweight="bold", color="#444444")
    ax.set_xlim(-16, 16); ax.set_xlabel("% of episodes"); ax.grid(axis="y", alpha=0)
    ax.legend(loc="lower left", fontsize=7, frameon=False, bbox_to_anchor=(0.0, -0.28), ncol=1)
    save(fig, "F3_rescues_harms"); plt.close(fig)


def f4():
    nc, cr = load("results/part5/nc/results.json"), load("results/part5/critic/results.json")
    cur = nc["A_R"]["curve"]
    ks = [1, 2, 4, 8, 16, 32, 64]
    est = [100 * cur[str(k)]["est"] for k in ks]
    lo = [100 * cur[str(k)]["ci"][0] for k in ks]
    hi = [100 * cur[str(k)]["ci"][1] for k in ks]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.fill_between(ks, lo, hi, color=C["sky"], alpha=0.3, lw=0)
    ax.plot(ks, est, "o-", color=C["blue"], ms=4, label="oracle: best of k noise draws")
    one = 100 * nc["A_R"]["mean_p"]["est"]
    ax.axhline(one, color=C["black"], ls="--", lw=0.9, label=f"one random retry ({one:.1f}%)")
    cks = [4, 8, 16]
    cv = [100 * cr["A"][f"k{k}"]["critic"] for k in cks]
    cl = [100 * cr["A"][f"k{k}"]["critic_ci"][0] for k in cks]
    ch = [100 * cr["A"][f"k{k}"]["critic_ci"][1] for k in cks]
    ax.errorbar(cks, cv, yerr=[np.array(cv) - cl, np.array(ch) - cv], fmt="s", color=C["verm"], ms=5, capsize=2, lw=0.8, label="learned critic: best of k (held-out)")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k, retry noise draws per state"); ax.set_ylabel("P0-failed states rescued, %"); ax.set_ylim(0, 85)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.set_title(f"{nc['A_R']['n_states']} P0-failed first-alarm states; 95% cluster-bootstrap bands", fontsize=8)
    save(fig, "F4_any_of_k"); plt.close(fig)


def f5():
    nc = load("results/part5/nc/results.json")
    p = np.array([r["p"] for r in nc["A_R"]["_rows"]])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    edges = np.linspace(0, 1, 11)
    zero = int((p == 0).sum())
    ax.bar(0.0, zero, width=0.04, color=C["verm"])
    pos = p[p > 0]
    h, _ = np.histogram(pos, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    colr = [C["orange"] if m < 0.5 else C["green"] for m in mids]
    ax.bar(mids, h, width=0.085, color=colr)
    from matplotlib.patches import Patch
    hs = [Patch(color=C["verm"], label=f"habit: never succeeds ({zero})"), Patch(color=C["orange"], label=f"sometimes: 0 < p < 0.5 ({int(((p > 0) & (p < 0.5)).sum())})"),
          Patch(color=C["green"], label=f"usually: p >= 0.5 ({int((p >= 0.5).sum())})")]
    ax.set_xlabel("p_s, share of 64 retry draws that succeed, per state"); ax.set_ylabel("states"); ax.legend(handles=hs, frameon=False, fontsize=7.5, loc="upper right")
    ax.set_xlim(-0.06, 1.0)
    save(fig, "F5_ps_distribution"); plt.close(fig)


def f6():
    cr = load("results/part5/critic/results.json")["firstnoise"]
    rows = sorted(cr["per_state"], key=lambda r: r["good"] - r["bad"])
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for k, r in enumerate(rows):
        ax.plot([100 * r["bad"], 100 * r["good"]], [k, k], color=C["grey"], lw=1.2, zorder=1)
    ax.scatter([100 * r["bad"] for r in rows], range(len(rows)), color=C["verm"], zorder=2, s=26, label="first noise from a failed P5.0 draw")
    ax.scatter([100 * r["good"] for r in rows], range(len(rows)), color=C["green"], zorder=2, s=26, label="first noise from a successful P5.0 draw")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"task {r['task']} / init {r['init']}" for r in rows], fontsize=7)
    ax.set_xlabel("episode success after the first retry decision, %"); ax.set_xlim(-3, 108); ax.grid(axis="y", alpha=0)
    ax.set_title(f"First-noise effect {100 * cr['effect']:+.1f} pp [{100 * cr['effect_ci'][0]:+.1f}, {100 * cr['effect_ci'][1]:+.1f}], {cr['n_pos_states']} states positive, {cr['n_neg_states']} negative\n({cr["n_branches"] // cr["n_states"] // 2} + {cr["n_branches"] // cr["n_states"] // 2} branches per state)", fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, -0.38))
    save(fig, "F6_first_noise"); plt.close(fig)


FIGS = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6}

if __name__ == "__main__":
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
    print("figures:", sorted(p.name for p in (PAPER / "figures").iterdir()))
