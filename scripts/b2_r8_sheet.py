"""R8: blind review sheet for the second reviewer (Pragnya). 40 rows = 8 from each Part 1.5 sheet (S-A ... S-E), shuffled, WITHOUT the AI's category and without
any hint of which sheet a row came from. Same 6 panels as C1. Writes results/part1/b2/r8/ (images + a CSV template to fill in) and, separately, the answer key
results/part1/b2/r8_key.csv (row id -> sheet, AI category, AI comment). Do not open the key before the review is done.
  python scripts/b2_r8_sheet.py"""
import cv2, numpy as np, pandas as pd
from pathlib import Path
from second_look.scene import ROOT
sys_path = None
import review_sheet as rs   # noqa (scripts/ is on sys.path when run as python scripts/b2_r8_sheet.py)
from second_look.p15 import Split

OUT = ROOT / "results/part1/b2/r8"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(8)
cats = pd.read_csv(ROOT / "results/part1/p15/notes/c1_review.csv").set_index("pair_id")
rows = []
for sheet in ("S-A_val_random", "S-B_pointing_misses", "S-C_policy_false_alarms", "S-D_rotation_misses", "S-E_heldout_misses"):
    d = pd.read_csv(ROOT / f"results/part1/p15/sheets/{sheet}.csv"); pick = rng.permutation(len(d))[:8]
    for i in pick: rows.append((sheet, d.iloc[int(i)]))
order = rng.permutation(len(rows))
splits = {}; imgs = []; key = []
for n, j in enumerate(order, 1):
    sheet, r = rows[int(j)]
    sp = splits.setdefault(r.split, Split(r.split, "deltadino_seed0"))
    im = rs.row_image(sp, int(r.score_index), r.split)
    # blind: replace the caption bar (it prints the model's probability and its CHANGED/same call) by one with the pair, the dataset label and the geometry only
    m_ = sp.meta.iloc[int(r.score_index)]
    lab = "GRAY" if m_["gray"] else ("changed" if m_["label"] else "unchanged")
    cap = (f"{m_['pair_id']} | {r.split} | task{m_['task']} | v1={lab} | trans {m_['max_trans_cm']:.1f}cm rot {m_['max_rot_deg']:.0f}deg | "
           f"items: {str(m_['changed_items'])[:48]} | d={m_['delta']} k={m_['k']}")
    im[:18] = 25
    cv2.putText(im, cap, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    imgs.append(im)
    c = cats.loc[r.pair_id]; key.append({"row_id": f"R8-{n:02d}", "pair_id": r.pair_id, "sheet": sheet, "ai_category": c.category, "ai_comment": c.comment})
# label each row with its id in the caption bar (replace the left part of the bar with the row id only: the caption already shows the pair id, split, generator, label, probability)
for n, im in enumerate(imgs, 1):
    cv2.putText(im, f"R8-{n:02d}", (im.shape[1] - 70, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
for s in range(0, len(imgs), 5):
    cv2.imwrite(str(OUT / f"review_{s // 5 + 1:02d}.png"), np.concatenate([rs.header(imgs[0].shape[1])] + imgs[s:s + 5], 0)[:, :, ::-1])
pd.DataFrame(key).to_csv(ROOT / "results/part1/b2/r8_key.csv", index=False)
tpl = pd.DataFrame({"row_id": [k["row_id"] for k in key], "pair_id": [k["pair_id"] for k in key], "category": "", "comment": ""})
tpl.to_csv(OUT / "r8_review_template.csv", index=False)
(OUT / "README.txt").write_text("""R8 second review (Pragnya)
40 rows in 8 images (review_01.png ... review_08.png, 5 rows each). Each row has 6 panels:
  ref | set | |set - ref| x4 | arm mask (red) | changed objects (green) | change map (hot = the model thinks 'changed'; green box = the hottest patch, cyan boxes = patches where the object really changed)
The caption above each row: pair id | split | source, generator, task | label (v1) and the model's probability | max translation / rotation | items | delta, k.
For EVERY row, pick ONE category and write it in r8_review_template.csv (column 'category'; 'comment' optional, one line):
  ok               prediction and label both look right
  label_invisible  the label says 'changed' but the change cannot be seen in the image (e.g. a spinning bowl, or a change fully behind the arm)
  label_wrong      the label looks wrong for another reason
  map_offset       the change map is hot NEXT TO the object (shadow, 1-patch ring, partly masked object)
  map_elsewhere    the change map is hot somewhere unrelated to any change
  arm_related      the decision seems driven by the arm, the gripper or a held object
  subtle_real      a real small change that the model correctly or nearly caught
  other            describe it in the comment
Please do not open results/part1/b2/r8_key.csv (it holds the AI's categories) until you are done.
""")
print("blind sheet:", len(imgs), "rows,", len(list(OUT.glob('review_*.png'))), "images ->", OUT)
