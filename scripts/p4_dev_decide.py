"""P4.6: choose delta (and tau) on DEV by the pre-written rule, apply the go/no-go (SPEC_part4.md section 7), write results/part4/notes/p4_dev_decision.{json,md}.

  python scripts/p4_dev_decide.py [--dev results/part4/dev] [--write]

DEV runs P0, P2r, PC (every alarmed episode carries the feature vector). Every SH variant is an EXACT selection between those two physical arms: all arms are bitwise P0 until the first alarm, the pause is shared,
the features and (p_retry, p_cont) depend only on the state at the decision, so SH with threshold delta behaves exactly like P2r when it retries and like PC when it continues (the TEST runs the SH arms live and the
analysis asserts this equality episode by episode). SH+F: flagged at the decision when max(p_retry, p_cont) < tau (the episode ends flagged), else SH.
Rule: delta in {0, .05, .10, .15, .20} maximising S on DEV (all 10 tasks), ties to the larger delta, separately for SH, SH-task, SH-noG; tau in {0.1, 0.2, 0.3} maximising S - SF subject to FF <= 5% (SH at its delta);
if none is feasible SH+F is dropped. GO iff S(SH) - S(P0) >= +2.0 pp AND S(SH) >= S(P2r).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C

ROOT = C.ROOT
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20]
TAUS = [0.1, 0.2, 0.3]


def episodes(recs, models):
    rows = []
    for k, r in recs.items():
        p0, p2r, pc = r["arms"]["P0"], C.arm_of(r, "P2r"), C.arm_of(r, "PC")
        ev = C.sh_event(p2r) if "P2r" in r["arms"] else None
        row = {"key": k, "task": k[0], "P0": p0, "P2r": p2r, "PC": pc, "x": None}
        if ev is not None:
            x = np.array(ev["x"], float)
            row["x"] = x
            row["p"] = {nm: m.predict(x) for nm, m in models.items()}
        rows.append(row)
    return rows


def outcome(row, model, delta, tau=None):
    """(success, flagged, retried) of SH(model, delta[, tau]) on an episode."""
    if row["x"] is None:
        a = row["P2r"]
        return bool(a["success"]), bool(a["flagged"]), False
    pr, pc = row["p"][model]
    if tau is not None and max(pr, pc) < tau:
        return False, True, False
    retry = (pr - pc) > delta
    a = row["P2r"] if retry else row["PC"]
    return bool(a["success"]), bool(a["flagged"]), retry


def metrics(rows, fn):
    n = len(rows)
    o = [fn(r) for r in rows]
    S = np.array([x[0] for x in o], float)
    F = np.array([x[1] for x in o], float)
    p0 = np.array([bool(r["P0"]["success"]) for r in rows], float)
    return {"n": n, "S": float(S.mean()), "SF": float(((1 - S) * (1 - F)).mean()), "F": float(F.mean()), "FF": float((F * p0).mean()), "retry_rate": float(np.mean([x[2] for x in o])),
            "rescues": int(((S == 1) & (p0 == 0)).sum()), "harms": int(((S == 0) & (p0 == 1)).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/part4/dev")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    blob = (ROOT / "configs/part4_model.pkl").read_bytes()
    models = pickle.loads(blob)
    rows = episodes(C.load(a.dev), models)
    seen = [r for r in rows if r["task"] not in C.HELD]
    res = {"n_episodes": len(rows), "model_sha256": hashlib.sha256(blob).hexdigest(), "P0": metrics(rows, lambda r: (bool(r["P0"]["success"]), False, False)),
           "P2r": metrics(rows, lambda r: (bool(r["P2r"]["success"]), bool(r["P2r"]["flagged"]), r["x"] is not None)), "PC": metrics(rows, lambda r: (bool(r["PC"]["success"]), bool(r["PC"]["flagged"]), False)), "grid": {}}
    alarmed = [r for r in rows if r["x"] is not None]
    res["n_alarmed_with_decision"] = len(alarmed)
    delta = {}
    for nm in ("sh", "task", "noG"):
        res["grid"][nm] = {str(d): metrics(rows, lambda r, d=d, nm=nm: outcome(r, nm, d)) for d in DELTAS}
        best = max(DELTAS, key=lambda d: (round(res["grid"][nm][str(d)]["S"], 9), d))                # max S, ties to the larger delta
        delta[nm] = best
    res["delta"] = delta
    res["SH_seen_at_delta"] = {nm: metrics(seen, lambda r, nm=nm: outcome(r, nm, delta[nm])) for nm in delta}
    sh = res["grid"]["sh"][str(delta["sh"])]
    res["tau_grid"] = {str(t): metrics(rows, lambda r, t=t: outcome(r, "sh", delta["sh"], t)) for t in TAUS}
    feas = [t for t in TAUS if res["tau_grid"][str(t)]["FF"] <= 0.05]
    tau = max(feas, key=lambda t: (res["tau_grid"][str(t)]["S"] - res["tau_grid"][str(t)]["SF"], -t)) if feas else None
    res["tau"] = tau
    res["go_no_go"] = {"S_SH_minus_S_P0_pp": 100 * (sh["S"] - res["P0"]["S"]), "S_SH_minus_S_P2r_pp": 100 * (sh["S"] - res["P2r"]["S"]), "criterion_1": bool(100 * (sh["S"] - res["P0"]["S"]) >= 2.0),
                       "criterion_2": bool(sh["S"] >= res["P2r"]["S"])}
    res["go"] = bool(res["go_no_go"]["criterion_1"] and res["go_no_go"]["criterion_2"])
    # ORACLE on DEV (one draw per branch: biased up; descriptive)
    res["oracle_dev"] = float(np.mean([max(bool(r["P2r"]["success"]), bool(r["PC"]["success"])) if r["x"] is not None else bool(r["P2r"]["success"]) for r in rows]))
    L = [f"# P4.6 DEV decision ({len(rows)} episodes, {len(alarmed)} alarmed with a decision; init 40-49 x p4dev-1,2; all 10 tasks)", "",
         f"P0 S {100 * res['P0']['S']:.1f}%, P2r (always retry) S {100 * res['P2r']['S']:.1f}% ({res['P2r']['rescues']} rescues / {res['P2r']['harms']} harms), PC (always continue) S {100 * res['PC']['S']:.1f}% ({res['PC']['rescues']} / {res['PC']['harms']}); "
         f"ORACLE (best of the two single draws, biased up) {100 * res['oracle_dev']:.1f}%.", "", "| model | delta | S % | retry rate | rescues | harms | SF % |", "|---|---|---|---|---|---|---|"]
    for nm, g in res["grid"].items():
        for d, m in g.items():
            L.append(f"| {nm} | {d}{' **chosen**' if float(d) == delta[nm] else ''} | {100 * m['S']:.1f} | {100 * m['retry_rate']:.0f}% | {m['rescues']} | {m['harms']} | {100 * m['SF']:.1f} |")
    L += ["", "SH+F (SH at its delta): " + "; ".join(f"tau {t}: S {100 * m['S']:.1f}, SF {100 * m['SF']:.1f}, FF {100 * m['FF']:.1f}%" for t, m in res["tau_grid"].items()) + f" -> tau = {tau} ({'dropped from TEST' if tau is None else 'chosen'})", "",
          f"**Go / no-go:** S(SH) - S(P0) = {res['go_no_go']['S_SH_minus_S_P0_pp']:+.1f} pp (needs >= +2.0): {res['go_no_go']['criterion_1']}; S(SH) - S(P2r) = {res['go_no_go']['S_SH_minus_S_P2r_pp']:+.1f} pp (needs >= 0): {res['go_no_go']['criterion_2']} "
          f"-> **{'GO: run TEST' if res['go'] else 'NO-GO: TEST is not run; Part 4 reports a DEV-level negative result'}**."]
    print("\n".join(L))
    if a.write:
        json.dump({"delta": delta, "tau": tau, "go": res["go"], "model_sha256": res["model_sha256"], "dev": a.dev, "summary": {k: res[k] for k in ("P0", "P2r", "PC", "go_no_go", "n_episodes", "n_alarmed_with_decision")},
                   "sh_at_delta": sh, "rule": "delta in {0,.05,.10,.15,.20}: max S on DEV all tasks, ties to the larger delta; tau in {.1,.2,.3}: max S-SF s.t. FF<=5%; go iff S(SH)-S(P0)>=2pp and S(SH)>=S(P2r)"},
                  open(ROOT / "results/part4/notes/p4_dev_decision.json", "w"), indent=1)
        (ROOT / "results/part4/notes/p4_dev_decision.md").write_text("\n".join(L))
        json.dump(res, open(ROOT / "results/part4/dev_report.json", "w"), indent=1, default=float)
        print("wrote results/part4/notes/p4_dev_decision.{json,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
