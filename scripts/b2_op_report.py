"""Operating-point table for the CHOSEN model's seeds (C9 format) -> results/part1/b2/notes/r6_operating_points.md.  python scripts/b2_op_report.py --json results/part1/b2/r6_operating_points.json"""
import argparse, json
from second_look.scene import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--title", default="chosen model"); a = ap.parse_args()
c = json.load(open(ROOT / a.json)); f = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"
SPL = (("val", "val"), ("test", "test"), ("test_heldout_tasks", "held-out"), ("test_policy", "policy"))
L = [f"# R6: operating points for Part 2 ({a.title})", "", "Thresholds are chosen **on val only** (`label_v2`, non-gray), per checkpoint, at val recall targets 0.95 / 0.98 / 0.99 / 0.995 (largest threshold reaching the target) and at F1-max, then frozen and applied to every split. Recall is on changed pairs; unchanged-flag = share of unchanged pairs called changed. **No threshold is chosen here: Part 2 chooses from this table using the costs of a missed change versus a skipped retry.** The sealed set `test_policy2` is not part of this table.", ""]
for m, r in c.items():
    L += [f"## {m}", "", "Thresholds (logit): " + ", ".join(f"{k} = {v:.2f}" for k, v in r["thresholds"].items()), "", "| operating point | " + " | ".join(f"{n}: recall / unchanged-flag / precision" for _, n in SPL) + " | big-change recall (val / test / held-out / policy) | H1+H2 FA (val / test / held-out) |", "|---|" + "---|" * (len(SPL) + 2)]
    for name, row in r["points"].items():
        L.append(f"| {name} | " + " | ".join(f"{f(row[s]['recall'])} / {f(row[s]['unchanged_flag_rate'])} / {f(row[s]['precision'])}" for s, _ in SPL) + " | " + " / ".join(f(row[s]["big_change_recall"]) for s, _ in SPL) + " | " + " / ".join(f(row[s]["h1h2_false_alarm"]) for s in ("val", "test", "test_heldout_tasks")) + " |")
    L.append("")
(ROOT / "results/part1/b2/notes/r6_operating_points.md").write_text("\n".join(L)); print("written")
