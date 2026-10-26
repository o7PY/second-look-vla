"""C1: review sheets. One row per pair, 6 panels:
   I_ref | I_set | |I_set-I_ref| x4 | arm mask (as used by the model) | changed-object mask (ref U set) | DeltaDINO change map on I_set
   (argmax patch outlined green, ground-truth changed patches outlined cyan).

  python scripts/review_sheet.py --all                      # the five C1 sheets (S-A .. S-E)
  python scripts/review_sheet.py --split test_policy --ids ID1 ID2 ... --name S-F_custom [--method deltadino_seed0]
Rows are read one at a time from the memory-mapped cache (never a whole shard)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from second_look.p15 import METHOD, Split, select_sheets
from second_look.scene import ROOT

P = 14                                  # patch size in px
W = 224
COLS = ("ref", "set", "diff x4", "arm mask", "changed objects", "change map")


def up(patches: np.ndarray) -> np.ndarray:
    return np.kron(patches.astype(np.uint8), np.ones((P, P), np.uint8)).astype(bool)


def outline(img, patchmask, color):
    for gy, gx in zip(*np.nonzero(patchmask)):
        cv2.rectangle(img, (gx * P, gy * P), (gx * P + P - 1, gy * P + P - 1), color, 1)


def row_image(sp: Split, i: int, split_label: str) -> np.ndarray:
    ref, st, m = sp.images(i)
    diff = np.clip(np.abs(st.astype(int) - ref.astype(int)) * 4, 0, 255).astype(np.uint8)
    armp = up(sp.arm[i])
    arm = (st * 0.35).astype(np.uint8)
    arm[armp] = (0.35 * st[armp] + 0.65 * np.array([255, 60, 60])).astype(np.uint8)
    chg_px = m[2] | m[3]
    ch = (st * 0.45).astype(np.uint8)
    ch[chg_px] = (0.4 * st[chg_px] + 0.6 * np.array([60, 255, 60])).astype(np.uint8)
    p = 1 / (1 + np.exp(-sp.pmap[i]))
    p = np.where(sp.arm[i], 0, p)
    heat = cv2.applyColorMap((np.clip(np.kron(p, np.ones((P, P))), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    cm = (0.5 * st + 0.5 * heat).astype(np.uint8)
    outline(cm, sp.chg[i], (0, 255, 255))
    a = sp.argmax_patch(i)
    if a is not None:
        am = np.zeros((16, 16), bool)
        am[a] = True
        outline(cm, am, (0, 255, 0))
    img = np.concatenate([ref, st, diff, arm, ch, cm], axis=1)
    m_ = sp.meta.iloc[i]
    lab = "GRAY" if m_["gray"] else ("changed" if m_["label"] else "unchanged")
    prob = 1 / (1 + np.exp(-sp.score[i]))
    cap = (f"{m_['pair_id']} | {split_label} | {m_['source']} {m_['gen']} task{m_['task']} | v1={lab} P={prob:.2f} ({'CHANGED' if sp.pred[i] else 'same'}) | "
           f"trans {m_['max_trans_cm']:.1f}cm rot {m_['max_rot_deg']:.0f}deg | items: {str(m_['changed_items'])[:48]} | d={m_['delta']} k={m_['k']}")
    bar = np.full((18, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, cap, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def header(width):
    h = np.full((16, width, 3), 60, np.uint8)
    for k, c in enumerate(COLS):
        cv2.putText(h, c, (k * W + 4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 160), 1, cv2.LINE_AA)
    return h


def render(name: str, rows: list[tuple[str, int]], method: str, out: Path, per_sheet: int = 6) -> list[dict]:
    splits: dict[str, Split] = {}
    images, listing = [], []
    for sp_name, i in rows:
        sp = splits.setdefault(sp_name, Split(sp_name, method))
        images.append(row_image(sp, i, sp_name))
        m = sp.meta.iloc[i]
        listing.append({"pair_id": m["pair_id"], "split": sp_name, "score_index": int(i), "gen": m["gen"], "task": int(m["task"]),
                        "label_v1": int(m["label"]), "pred": int(sp.pred[i]), "p": float(1 / (1 + np.exp(-sp.score[i]))),
                        "max_trans_cm": float(m["max_trans_cm"]), "max_rot_deg": float(m["max_rot_deg"]),
                        "pointing_hit": sp.pointing_hit(i)})
    out.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(images), per_sheet):
        chunk = images[s:s + per_sheet]
        cv2.imwrite(str(out / f"{name}_{s // per_sheet:02d}.png"), np.concatenate([header(chunk[0].shape[1])] + chunk, axis=0)[:, :, ::-1])
    pd.DataFrame(listing).to_csv(out / f"{name}.csv", index=False)
    print(f"{name}: {len(images)} rows -> {out} ({(len(images) + per_sheet - 1) // per_sheet} images)", flush=True)
    return listing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true"); ap.add_argument("--split"); ap.add_argument("--ids", nargs="*")
    ap.add_argument("--name", default="custom"); ap.add_argument("--method", default=METHOD)
    ap.add_argument("--out", default=str(ROOT / "results/part1/p15/sheets"))
    a = ap.parse_args()
    out = Path(a.out)
    if a.all:
        sel = select_sheets()
        sel.pop("_pool_sizes", None)
        for name, rows in sel.items():
            render(name, rows, a.method, out)
    else:
        sp = Split(a.split, a.method)
        idx = {p: k for k, p in enumerate(sp.meta["pair_id"])}
        render(a.name, [(a.split, idx[p]) for p in a.ids], a.method, out)
