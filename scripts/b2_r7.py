"""R7: one-shot final evaluation on the SEALED set test_policy2 (SPEC_before_part2.md section 11). For the chosen model (3 seeds) and the other candidate (context only,
marked not chosen), each seed at its F1-max val threshold and at its val-recall-0.99 threshold (both chosen on val with label_v2, before this script is run):
AUROC (95% CI by bootstrap over episodes, seed mean), F1, precision, recall, big-change recall, unchanged-flag rate, gray call rate; P_adj, P_visible;
seen vs held-out tasks (6, 8); successful vs failed episodes and their intersections; per task. Removal-drop strata come from scripts/b2_r1.py (--splits test_policy2).
RULE: no model, threshold, mask or label change after seeing these numbers. Refuses to run on test_policy2 unless results/part1/b2/notes/r6_decision.md is committed (src/second_look/sealed.py).
  python scripts/b2_r7.py                       # the real thing
  python scripts/b2_r7.py --split test_policy   # dry run on a DEV set to debug the script (writes *_dryrun files)"""
import argparse, json
import numpy as np, pandas as pd
from second_look import metrics as M, p15
from second_look.eval import load_model, score_model
from second_look.p15 import Split
from second_look.scene import ROOT
from second_look.sealed import check_sealed

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test_policy2"); ap.add_argument("--n_boot", type=int, default=1000); a = ap.parse_args()
check_sealed(a.split)
OUT = ROOT / "results/part1/b2"; suf = "" if a.split == "test_policy2" else "_dryrun"
r6 = json.load(open(OUT / "r6.json")); chosen = r6["choice"]["model"]; other = "C" if chosen == "A" else "A"
CAND = {"A": ("deltadino_v2_seed", "results/part1/deltadino_v2/seed{}/best.pt"), "C": ("deltadino_v2static_seed", "results/part1/deltadino_v2_static/seed{}/best.pt")}
HELD = (6, 8); rng = np.random.default_rng(0)


def ensure_scores(method, ckpt):
    if not (ROOT / "results/part1/scores" / method / f"{a.split}.npz").exists():
        model, _ = load_model(ckpt); M.save_scores(method, a.split, **score_model(model, a.split)); del model


def thr_recall(method, target=0.99):
    v = Split("val", method, thr=0.0, labels="v2"); ng = v.nongray; pos = np.sort(v.score[ng & (v.y == 1)])
    return float(pos[int(np.floor((1 - target) * len(pos)))])


def metrics(S: Split, mask=None):
    m = S.nongray if mask is None else (S.nongray & mask)
    y, s, pred = S.y[m], S.score[m], S.pred[m]
    o = {"n": int(m.sum()), "pos": int(y.sum()), "auroc": M.auroc(y, s) if len(np.unique(y)) == 2 else None}
    P = M.prf(y, s, S.thr); o.update({k: P[k] for k in ("f1", "precision", "recall")})
    big = m & (S.y == 1) & ((S.meta.max_trans_cm.values > 2) | (S.meta.max_rot_deg.values > 15))
    o["big_change_recall"] = float(S.pred[big].mean()) if big.any() else None
    o["unchanged_flag"] = float(pred[y == 0].mean()) if (y == 0).any() else None
    g = S.meta.gray.values.astype(bool) & (np.ones(len(m), bool) if mask is None else mask)
    o["gray_call_rate"] = float(S.pred[g].mean()) if g.any() else None; o["gray_n"] = int(g.sum())
    det = np.flatnonzero(m & (S.y == 1) & S.pred); rows = [p15.pointing_row(S, int(i)) for i in det]
    padj = [r["P_adj"] for r in rows if r["P_adj"] is not None]; pvis = [r["P_visible"] for r in rows if r["P_visible"] is not None]
    o["P_adj"] = float(np.mean(padj)) if padj else None; o["P_visible"] = float(np.mean(pvis)) if pvis else None
    return o


def boot_auroc(Ss, mask=None, n=None):
    n = n or a.n_boot; S0 = Ss[0]; m = S0.nongray if mask is None else (S0.nongray & mask)
    meta = S0.meta[m].reset_index(drop=True); y = S0.y[m]; ep = (meta.task.astype(int) * 1000 + meta.demo.astype(int)).values
    u, inv = np.unique(ep, return_inverse=True); idx = [np.flatnonzero(inv == j) for j in range(len(u))]; sc = [S.score[m] for S in Ss]; v = []
    for _ in range(n):
        ii = np.concatenate([idx[p] for p in rng.integers(len(u), size=len(u))])
        if len(np.unique(y[ii])) < 2: continue
        v.append(np.mean([M.auroc(y[ii], s[ii]) for s in sc]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]


res = {"split": a.split, "chosen": chosen, "other": other, "models": {}}
for key in (chosen, other):
    prefix, ck = CAND[key]; methods = [f"{prefix}{k}" for k in range(3)]
    for k, m in enumerate(methods): ensure_scores(m, ck.format(k))
    entry = {"role": "chosen" if key == chosen else "not chosen (context only)", "seeds": {}}
    for rule in ("F1-max", "val recall 0.99"):
        Ss = []
        for m in methods:
            thr = Split("val", m, labels="v2").thr if rule == "F1-max" else thr_recall(m)
            Ss.append(Split(a.split, m, labels="v2", thr=thr))
        task = Ss[0].meta.task.astype(int).values; succ = Ss[0].meta.success.astype(bool).values; held = np.isin(task, HELD)
        groups = {"all": None, "seen tasks": ~held, "held-out tasks (6, 8)": held, "successful episodes": succ, "failed episodes": ~succ,
                  "seen + successful": ~held & succ, "seen + failed": ~held & ~succ, "held-out + successful": held & succ, "held-out + failed (worst case)": held & ~succ}
        per_group = {}
        for g, mk in groups.items():
            ms = [metrics(S, mk) for S in Ss]
            agg = {k: (float(np.mean([x[k] for x in ms if x[k] is not None])) if any(x[k] is not None for x in ms) else None) for k in ("auroc", "f1", "precision", "recall", "big_change_recall", "unchanged_flag", "gray_call_rate", "P_adj", "P_visible")}
            agg.update(n=ms[0]["n"], pos=ms[0]["pos"], gray_n=ms[0]["gray_n"], per_seed=ms)
            if g in ("all", "seen tasks", "held-out tasks (6, 8)", "failed episodes"): agg["auroc_ci"] = boot_auroc(Ss, mk)
            per_group[g] = agg
        per_task = {int(t): {k: v for k, v in metrics_ .items() if k != "per_seed"} for t in sorted(set(task)) for metrics_ in [{**{k: float(np.mean([x[k] for x in (metrics(S, task == t) for S in Ss) if x[k] is not None])) if any(x[k] is not None for x in [metrics(S, task == t) for S in Ss]) else None for k in ("auroc", "recall", "big_change_recall", "unchanged_flag", "P_adj")}, "n": int(((task == t) & Ss[0].nongray).sum())}]}
        entry[rule] = {"thresholds": [S.thr for S in Ss], "groups": per_group, "per_task": per_task}
    res["models"][key] = entry
    print(key, "done", flush=True)
json.dump(res, open(OUT / f"test_policy2_eval{suf}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote", OUT / f"test_policy2_eval{suf}.json")
