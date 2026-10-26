"""P2.0: operating-point table of model C (static mask only), val thresholds (label_v2), 3 seeds -> results/part2/notes/c_operating_points.md.
  python scripts/c9_operating_points.py --methods deltadino_v2static_seed0 deltadino_v2static_seed1 deltadino_v2static_seed2 --out results/part2/c_operating_points.json
  python scripts/p2_op_report.py
Same format as results/part1/b2/notes/r6_operating_points.md (A). The sealed set test_policy2 is not part of this table; no threshold is chosen here (P2.5 chooses on DEV)."""
import json
from second_look.scene import ROOT

c = json.load(open(ROOT / "results/part2/c_operating_points.json")); f = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"
SPL = (("val", "val"), ("test", "test"), ("test_heldout_tasks", "held-out"), ("test_policy", "policy"))
L = ["# P2.0: operating points of model C (static mask only, DeltaDINO-v2-static, 3 seeds)", "",
     "Thresholds are chosen **on val only** (`label_v2`, non-gray), per checkpoint, at F1-max and at val recall 0.95 / 0.98 / 0.99 / 0.995 (largest threshold reaching the target), then frozen and applied to every split. "
     "Recall is on changed pairs; unchanged-flag = share of unchanged pairs called changed. Same procedure and format as `results/part1/b2/notes/r6_operating_points.md` (model A). "
     "`test_policy2` is not part of this table. No threshold is chosen here; P2.5 chooses on DEV.", ""]
for m, r in c.items():
    L += [f"## {m}", "", "Thresholds (logit): " + ", ".join(f"{k} = {v:.2f}" for k, v in r["thresholds"].items()), "",
          "| operating point | " + " | ".join(f"{n}: recall / unchanged-flag / precision" for _, n in SPL) + " | big-change recall (val / test / held-out / policy) | H1+H2 FA (val / test / held-out) |", "|---|" + "---|" * (len(SPL) + 2)]
    for name, row in r["points"].items():
        L.append(f"| {name} | " + " | ".join(f"{f(row[s]['recall'])} / {f(row[s]['unchanged_flag_rate'])} / {f(row[s]['precision'])}" for s, _ in SPL) + " | "
                 + " / ".join(f(row[s]["big_change_recall"]) for s, _ in SPL) + " | " + " / ".join(f(row[s]["h1h2_false_alarm"]) for s in ("val", "test", "test_heldout_tasks")) + " |")
    L.append("")
(ROOT / "results/part2/notes/c_operating_points.md").write_text("\n".join(L)); print("written results/part2/notes/c_operating_points.md")
