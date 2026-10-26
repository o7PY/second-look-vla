"""R8: agreement between the second reviewer (Pragnya) and the AI's C1 categories on the 40 blind rows.
  python scripts/b2_r8_score.py --review results/part1/b2/r8/r8_review_template.csv   (filled in by the reviewer; the file the reviewer returns)
Writes results/part1/b2/notes/r8_review.csv (row, pair, both categories, agree) and results/part1/b2/notes/r8_human_review.md (raw agreement, Cohen's kappa overall and per sheet,
confusion table, every disagreement with both categories and the reviewer's comment). The AI key (results/part1/b2/r8_key.csv) is only read here."""
import argparse
import numpy as np, pandas as pd
from second_look.scene import ROOT

CATS = ["ok", "label_invisible", "label_wrong", "map_offset", "map_elsewhere", "arm_related", "subtle_real", "other"]
ap = argparse.ArgumentParser(); ap.add_argument("--review", required=True); ap.add_argument("--out_md", default=str(ROOT / "results/part1/b2/notes/r8_human_review.md")); ap.add_argument("--out_csv", default=str(ROOT / "results/part1/b2/notes/r8_review.csv")); a = ap.parse_args()
rev = pd.read_csv(a.review); key = pd.read_csv(ROOT / "results/part1/b2/r8_key.csv")
rev["category"] = rev["category"].astype(str).str.strip().str.lower(); rev.loc[rev.category.isin(["nan", ""]), "category"] = np.nan
missing = rev[rev.category.isna()].row_id.tolist(); bad = rev[rev.category.notna() & ~rev.category.isin(CATS)]
assert not missing, f"unfilled rows: {missing}"
assert bad.empty, f"unknown categories: {bad[['row_id','category']].values.tolist()}"
d = key.merge(rev[["row_id", "category", "comment"]], on="row_id", suffixes=("_ai", "")).rename(columns={"category": "human_category", "comment": "human_comment", "ai_category": "ai_category"})
d["agree"] = d.ai_category == d.human_category
def kappa(x, y):
    n = len(x); po = float(np.mean(np.array(x) == np.array(y)))
    pe = sum((np.mean(np.array(x) == c)) * (np.mean(np.array(y) == c)) for c in CATS)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
k_all = kappa(d.ai_category, d.human_category)
L = ["# R8: second reviewer (Pragnya) on 40 blind rows", "",
     "40 rows, 8 from each Part 1.5 sheet (S-A … S-E), shuffled, the AI's category hidden (`scripts/b2_r8_sheet.py`, `results/part1/b2/r8/`). Both reviewers used the C1 categories. The AI's categories were made earlier by looking at every row of the five sheets (`results/part1/p15/notes/c1_review.csv`).", "",
     f"- **Raw agreement: {d.agree.mean():.1%} ({int(d.agree.sum())} of {len(d)} rows).**", f"- **Cohen's κ: {k_all:.2f}** (8 categories; 0.41–0.60 is usually called moderate, 0.61–0.80 substantial; I am quoting the usual convention, not a rule from the spec).", "",
     "| sheet | rows | agreement | κ |", "|---|---|---|---|"]
for sh, g in d.groupby("sheet"):
    L.append(f"| {sh} | {len(g)} | {g.agree.mean():.0%} | {kappa(g.ai_category, g.human_category):.2f} |")
cm = pd.crosstab(d.ai_category, d.human_category).reindex(index=CATS, columns=CATS, fill_value=0)
L += ["", "Confusion table (rows = AI, columns = reviewer):", "", "| AI \\ reviewer | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)] + [f"| {c} | " + " | ".join(str(cm.loc[c, k]) for k in CATS) + " |" for c in CATS]
L += ["", "## Disagreements", "", "| row | sheet | pair | AI | reviewer | AI comment | reviewer comment |", "|---|---|---|---|---|---|---|"]
for _, r in d[~d.agree].iterrows():
    L.append(f"| {r.row_id} | {r.sheet} | `{r.pair_id}` | {r.ai_category} | {r.human_category} | {str(r.ai_comment)[:90]} | {'' if pd.isna(r.human_comment) else str(r.human_comment)[:90]} |")
L.append("")
open(a.out_md, "w").write("\n".join(L)); d.to_csv(a.out_csv, index=False); print(f"agreement {d.agree.mean():.3f}, kappa {k_all:.3f}, disagreements {int((~d.agree).sum())}")
