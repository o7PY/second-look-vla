"""Score a checkpoint on the given splits and save results/part1/scores/<method>/<split>.npz (does not recompute or overwrite other splits).
  python scripts/b2_score.py --ckpt results/part1/deltadino_v2_static/seed0/best.pt --method deltadino_v2static_seed0 --splits val test test_heldout_tasks test_policy probe_static
  Refuses sealed splits through second_look.sealed (PairDataset)."""
import argparse
from second_look import metrics as M
from second_look.eval import load_model, score_model
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--method", required=True); ap.add_argument("--splits", nargs="+", required=True)
a = ap.parse_args()
model, ck = load_model(a.ckpt)
for sp in a.splits:
    M.save_scores(a.method, sp, **score_model(model, sp)); print("scored", a.method, sp, flush=True)
