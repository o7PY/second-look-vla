"""P4 checks: counts vs SPEC §4.5 (±5%), no episode in two splits, per-split stats, disk use.
   python scripts/check_dataset.py [--pairs data/pairs]  -> results/part1/notes/dataset_stats.md"""
import argparse, json, shutil, subprocess
import pandas as pd
from second_look.scene import ROOT
from second_look.sampler import SPLITS_PATH, splits_sha256

ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default="data/pairs"); a = ap.parse_args()
P = ROOT / a.pairs
sp = json.loads(SPLITS_PATH.read_text())["episodes"]
metas = {s: pd.read_parquet(P / s / "meta.parquet").assign(split=s) for s in ("train", "val", "test", "test_heldout_tasks") if (P / s / "meta.parquet").exists()}
allm = pd.concat(metas.values(), ignore_index=True)
allm["ep"] = allm["task"].astype(str) + ":" + allm["demo"].astype(str)
# no episode in two splits, and every episode is in the split splits.json assigned
chk = allm.groupby("ep")["split"].nunique()
assert (chk == 1).all(), f"episodes in >1 split: {list(chk[chk > 1].index)[:5]}"
bad = [e for e, s in zip(allm["ep"], allm["split"]) if sp[e] != s]
assert not bad, f"episode/split mismatch vs splits.json: {bad[:5]}"
print("OK: no episode appears in two splits; assignments match splits.json")

ng = allm[~allm.gray]
rows = ["| split | episodes | pairs | unchanged | changed | gray | S1 | S2 |", "|---|---|---|---|---|---|---|---|"]
for s, m in metas.items():
    mn = m[~m.gray]
    rows.append(f"| {s} | {m.assign(ep=m.task.astype(str)+':'+m.demo.astype(str)).ep.nunique()} | {len(m)} | {int((mn.label==0).sum())} | {int((mn.label==1).sum())} | {int(m.gray.sum())} | {int((m.source=='S1').sum())} | {int((m.source=='S2').sum())} |")
tr_val_te = ng[ng.split.isin(["train", "val", "test"])]
tot = len(allm); n0, n1 = int((ng.label == 0).sum()), int((ng.label == 1).sum())
rows.append(f"| **all** | {allm.ep.nunique()} | {tot} | {n0} | {n1} | {int(allm.gray.sum())} | {int((allm.source=='S1').sum())} | {int((allm.source=='S2').sum())} |")
gen = ng.groupby(["gen", "label"]).size().unstack(fill_value=0)
du = subprocess.check_output(["du", "-sh", str(P)]).decode().split()[0]
free = shutil.disk_usage(ROOT).free / 1e9
spec_ok = abs(tot / 60000 - 1) <= 0.05 and abs(n0 / 30000 - 1) <= 0.05 and abs(n1 / 30000 - 1) <= 0.05
txt = f"""# Dataset statistics (P4)

splits.json sha256: `{splits_sha256()}`

{chr(10).join(rows)}

Spec target ≈ 60k pairs, 30k unchanged / 30k changed (±5%): **{'met' if spec_ok else 'NOT met'}** (total {tot}, unchanged {n0}, changed {n1}, non-gray).
Gray-zone pairs are only generated for test splits (kept, never trained/validated on).

Per generator (non-gray):

{gen.to_markdown()}

Disk: `{a.pairs}` = {du}; free disk after generation: {free:.1f} GB.
"""
(ROOT / "results" / "part1" / "notes" / ("dataset_stats.md" if a.pairs == "data/pairs" else "dataset_stats_trial.md")).write_text(txt)
print(txt)
