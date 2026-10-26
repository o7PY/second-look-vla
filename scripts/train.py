"""python scripts/train.py --config configs/deltadino.yaml --seed 0 [--out DIR] [--set train.lam=0 model.lora_rank=16]"""
import argparse, json
from second_look.train import load_config, run
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out"); ap.add_argument("--max_steps", type=int); ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
print(json.dumps(cfg))
print(run(cfg, a.seed, a.out, a.max_steps))
