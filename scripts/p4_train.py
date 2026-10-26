"""P4.5: train and select the retry-value models on GEN only, by the pre-written criterion (SPEC_part4.md section 5); freeze the model.

  python scripts/p4_train.py [--gen results/part4/gen] [--write]        -> results/part4/notes/p4_5_model_report.md, results/part4/model_report.json; --write also configs/part4_model.{pkl,json}

Cases = GEN alarm cases with a feature vector (see p4_common.gen_cases). Rows = one per branch outcome (4 retry + 4 continue per case). T-learner: f_retry and f_cont, one classifier each, over the columns of the
group set: SH = T+A+D+G+R, SH-task = T only, SH-noG = T+A+D+R. Candidates: (i) L2-logistic on standardised features, (ii) gradient-boosted trees (depth <= 3, <= 200 trees) with isotonic calibration (cross-fitted by group).
Selection per model: the highest cross-validated DECISION VALUE = the mean, over held-out cases, of the branch success rate of the branch the rule "retry iff p_retry - p_cont > 0" picks (k_r / 4 or k_c / 4);
5-fold cross-validation grouped by (task, init), averaged over 5 shuffles of the group assignment (seed 0..4; variance reduction only); ties go to the logistic regression.
Reported: always-retry, always-continue, P0 (the same cases), ORACLE (best of the 4 + 4 branch draws, biased UP because it picks the maximum of two noisy estimates) and a split-half oracle (pick the better branch
on draws 1-2, score on draws 3-4: the same bias removed, not the information), AUROC of p_retry - p_cont for "retry better than continue" (k_r > k_c vs k_r < k_c), calibration (reliability table of the out-of-fold p),
feature importances. STOP rule: if the cross-validated decision value of SH is not above max(always-retry, always-continue) nothing is frozen (exit code 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C
from second_look import features as F
from second_look.shmodel import GBTCalibrated, Logistic, SHModel

ROOT = C.ROOT
MODELS = {"sh": "TADGR", "task": "T", "noG": "TADR"}
KINDS = ["logistic", "gbt"]


def make(kind):
    return Logistic() if kind == "logistic" else GBTCalibrated()


def fit_heads(kind, X, yr, yc, groups):
    """X (n_cases, p), yr / yc (n_cases, 4): one row per branch."""
    Xr, Xc = np.repeat(X, 4, axis=0), np.repeat(X, 4, axis=0)
    g = np.repeat(groups, 4)
    return make(kind).fit(Xr, yr.reshape(-1), g), make(kind).fit(Xc, yc.reshape(-1), g)


def predict(heads, X):
    return heads[0].predict_proba(X)[:, 1], heads[1].predict_proba(X)[:, 1]


def cv_predict(kind, X, yr, yc, groups, seed):
    ug = np.unique(groups)
    perm = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug))))
    g2 = np.array([perm[g] for g in groups])
    pr, pc = np.zeros(len(X)), np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=g2):
        heads = fit_heads(kind, X[tr], yr[tr], yc[tr], groups[tr])
        pr[te], pc[te] = predict(heads, X[te])
    return pr, pc


def decision_value(pr, pc, yr, yc, delta=0.0):
    retry = (pr - pc) > delta
    return float(np.where(retry, yr.mean(1), yc.mean(1)).mean()), float(retry.mean())


def calibration(p, y, bins=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001)):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            out.append({"bin": [lo, min(hi, 1.0)], "n": int(m.sum()), "mean_p": float(p[m].mean()), "frac_success": float(y[m].mean())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="results/part4/gen")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--shuffles", type=int, default=5)
    a = ap.parse_args()
    cases, skipped = C.gen_cases(C.load(a.gen))
    X = np.stack([c["x"] for c in cases])
    yr, yc = np.stack([c["y_r"] for c in cases]), np.stack([c["y_c"] for c in cases])
    groups = np.array([c["task"] * 100 + c["init"] for c in cases])
    p0 = np.array([c["p0"] for c in cases], float)
    n = len(cases)
    kr, kc = yr.mean(1), yc.mean(1)
    rep = {"n_cases": n, "skipped": skipped, "n_groups": int(len(np.unique(groups))), "mean_p_retry": float(kr.mean()), "mean_p_cont": float(kc.mean()), "P0_success_on_these_cases": float(p0.mean()),
           "oracle_max_of_4_and_4": float(np.maximum(kr, kc).mean()),
           "oracle_split_half": float(np.where(yr[:, :2].mean(1) > yc[:, :2].mean(1), yr[:, 2:].mean(1), np.where(yr[:, :2].mean(1) < yc[:, :2].mean(1), yc[:, 2:].mean(1), (yr[:, 2:].mean(1) + yc[:, 2:].mean(1)) / 2)).mean()),
           "by_task": {int(t): {"n": int((np.array([c['task'] for c in cases]) == t).sum()), "p_retry": float(kr[np.array([c['task'] for c in cases]) == t].mean()), "p_cont": float(kc[np.array([c['task'] for c in cases]) == t].mean())}
                       for t in sorted({c['task'] for c in cases})},
           "retry_better_cases": int((kr > kc).sum()), "continue_better_cases": int((kr < kc).sum()), "ties": int((kr == kc).sum()), "models": {}}
    chosen = {}
    for name, groups_ in MODELS.items():
        cols = F.columns(groups_)
        res = {}
        for kind in KINDS:
            dvs, rts, aucs, oof = [], [], [], []
            for s in range(a.shuffles):
                pr, pc = cv_predict(kind, X[:, cols], yr, yc, groups, s)
                dv, rt = decision_value(pr, pc, yr, yc)
                dvs.append(dv), rts.append(rt), oof.append((pr, pc))
                m = kr != kc
                aucs.append(float(roc_auc_score((kr > kc)[m], (pr - pc)[m])) if m.any() and len(np.unique((kr > kc)[m])) == 2 else None)
            res[kind] = {"cv_decision_value": float(np.mean(dvs)), "cv_decision_value_sd_over_shuffles": float(np.std(dvs)), "retry_rate": float(np.mean(rts)),
                         "auroc_retry_better": None if None in aucs else float(np.mean(aucs)), "oof": oof}
        best = max(KINDS, key=lambda k: (res[k]["cv_decision_value"], k == "logistic"))      # ties go to the logistic regression
        chosen[name] = best
        pr = np.mean([o[0] for o in res[best]["oof"]], 0)
        pc = np.mean([o[1] for o in res[best]["oof"]], 0)
        rep["models"][name] = {"groups": groups_, "chosen": best, "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in res.items()},
                               "calibration_retry_head": calibration(pr, yr.reshape(n, 4).mean(1)), "calibration_continue_head": calibration(pc, yc.mean(1))}
    rep["always_retry"], rep["always_continue"] = float(kr.mean()), float(kc.mean())
    sh = rep["models"]["sh"]["candidates"][chosen["sh"]]["cv_decision_value"]
    rep["sh_cv_decision_value"] = sh
    rep["passes_stop_rule"] = bool(sh > max(rep["always_retry"], rep["always_continue"]))
    # feature importances of the final fits (all cases)
    heads = {nm: fit_heads(chosen[nm], X[:, F.columns(g)], yr, yc, groups) for nm, g in MODELS.items()}
    names = lambda g: [F.FEATURE_NAMES[i] for i in F.columns(g)]
    for nm, g in MODELS.items():
        imp = {}
        for head, h in zip(("retry", "continue"), heads[nm]):
            v = h.coef_ if hasattr(h, "coef_") else h.feature_importances_
            if v is not None:
                imp[head] = sorted(zip(names(g), map(float, v)), key=lambda t: -abs(t[1]))[:10]
        rep["models"][nm]["top_features"] = imp
    out = ROOT / "results" / "part4"
    out.mkdir(exist_ok=True)
    json.dump(rep, open(out / "model_report.json", "w"), indent=1, default=float)
    L = [f"# P4.5 model report (GEN only; {n} alarm cases, {rep['n_groups']} (task, init) groups; skipped {skipped})", "",
         f"Branch means over the cases: always-retry {rep['always_retry']:.3f}, always-continue {rep['always_continue']:.3f}, P0 (no pause) on the same cases {rep['P0_success_on_these_cases']:.3f}. Cases where retry was better: {rep['retry_better_cases']}, "
         f"continue better: {rep['continue_better_cases']}, ties {rep['ties']}.", "",
         f"ORACLE (best of the 4 + 4 draws, biased up): {rep['oracle_max_of_4_and_4']:.3f}; split-half oracle (bias removed): {rep['oracle_split_half']:.3f}.", "",
         "| model | groups | candidate | CV decision value | sd over shuffles | retry rate | AUROC (retry better) | chosen |", "|---|---|---|---|---|---|---|---|"]
    for nm, m in rep["models"].items():
        for k, v in m["candidates"].items():
            L.append(f"| {nm} | {m['groups']} | {k} | {v['cv_decision_value']:.3f} | {v['cv_decision_value_sd_over_shuffles']:.3f} | {v['retry_rate']:.2f} | {v['auroc_retry_better'] if v['auroc_retry_better'] is None else round(v['auroc_retry_better'], 3)} | {'**yes**' if k == m['chosen'] else ''} |")
    L += ["", f"**Stop rule:** SH's cross-validated decision value {sh:.3f} vs max(always-retry, always-continue) {max(rep['always_retry'], rep['always_continue']):.3f}: **{'passes' if rep['passes_stop_rule'] else 'FAILS: nothing is frozen, Part 4 stops here'}**.", "",
          "Per task (cases, mean retry-branch success, mean continue-branch success): " + "; ".join(f"{t}: n={v['n']}, {v['p_retry']:.2f} / {v['p_cont']:.2f}" for t, v in rep["by_task"].items()), ""]
    for nm, m in rep["models"].items():
        L += [f"## {nm}: top features", ""] + [f"- {h}: " + ", ".join(f"{a_} {b:+.2f}" for a_, b in v[:6]) for h, v in m["top_features"].items()] + [""]
    (ROOT / "results" / "part4" / "notes" / "p4_5_model_report.md").write_text("\n".join(L))
    print("\n".join(L[:20]))
    if not rep["passes_stop_rule"]:
        return 3
    if a.write:
        models = {nm: SHModel(nm, g, chosen[nm], *heads[nm]) for nm, g in MODELS.items()}
        blob = pickle.dumps(models)
        (ROOT / "configs" / "part4_model.pkl").write_bytes(blob)
        json.dump({"sha256": hashlib.sha256(blob).hexdigest(), "feature_code_sha256": hashlib.sha256((ROOT / "src" / "second_look" / "features.py").read_bytes()).hexdigest(), "feature_names": F.FEATURE_NAMES,
                   "chosen": chosen, "n_cases": n, "cv_decision_values": {nm: rep["models"][nm]["candidates"][chosen[nm]]["cv_decision_value"] for nm in MODELS}, "always_retry": rep["always_retry"],
                   "always_continue": rep["always_continue"], "gen_dir": a.gen}, open(ROOT / "configs" / "part4_model.json", "w"), indent=1)
        print("wrote configs/part4_model.pkl and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
