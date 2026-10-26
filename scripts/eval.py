"""Score a checkpoint on splits and write metrics.  The threshold comes from val and is then frozen.

  python scripts/eval.py --ckpt results/part1/deltadino/seed0/best.pt --splits val test test_heldout_tasks test_policy
"""
import argparse, json
from pathlib import Path
import torch
from second_look import metrics as M
from second_look.eval import json_default, latency_ms, load_model, report_from_scores, score_model, val_threshold

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--splits", nargs="+", default=["val"])
ap.add_argument("--method"); ap.add_argument("--n_boot", type=int, default=1000)
a = ap.parse_args()
ck = Path(a.ckpt)
method = a.method or f"{ck.parent.parent.name}_{ck.parent.name}"
model, meta_ck = load_model(a.ckpt)
print(f"{method}: epoch {meta_ck['epoch']} val_auroc {meta_ck['val_auroc']:.4f} git {meta_ck['git']} seed {meta_ck['seed']}")
order = ["val"] + [s for s in a.splits if s != "val"]      # val first: it defines the threshold
res, thr = {}, None
for sp in order:
    sc = score_model(model, sp)
    M.save_scores(method, sp, **sc)
    if sp == "val":
        thr = val_threshold(method)
        print(f"frozen threshold (val F1-max, logit): {thr:.4f}")
    if sp not in a.splits:
        continue
    rep = report_from_scores(method, sp, thr, a.n_boot)
    res[sp] = rep
    i, hn = rep["image"], rep["hard_neg"]["H1+H2"]
    print(f"[{sp}] AUROC {i['auroc']:.4f} [{i['auroc_ci'][0]:.3f},{i['auroc_ci'][1]:.3f}]  AP {i['ap']:.4f}  F1 {i['f1']:.4f}  P {i['precision']:.3f} R {i['recall']:.3f} | "
          f"H1+H2 FAR {hn['false_alarm']:.3f} | recall(>2cm/15deg) {rep['big_change_recall']['recall']:.3f} | patch AUROC {rep['patch']['patch_auroc']:.3f} point {rep['patch']['pointing_acc']:.3f}")
res["latency_ms_batch1"] = latency_ms(model)
print(f"latency: {res['latency_ms_batch1']:.1f} ms/pair at batch 1")
out = ck.parent / "eval.json"
prev = json.load(open(out)) if out.exists() else {}
prev.update(res)
json.dump(prev, open(out, "w"), indent=1, default=json_default)
print("wrote", out)
