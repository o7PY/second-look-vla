"""S3 pass 2: pairs from the recorded policy rollouts (S1-style sampling, source S3) -> data/pairs/test_policy.
  python scripts/make_policy_pairs.py [--rollouts results/part1/policy_rollouts] [--per_class 10]
"""
import argparse, json, subprocess, sys, shutil
from pathlib import Path
import numpy as np
from second_look.scene import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", default="results/part1/policy_rollouts"); ap.add_argument("--per_class", type=int, default=10)
ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()

from second_look import render
from second_look.sampler import Demo, Quota, sample_episode
from second_look.scene import Scene
tmp = ROOT / "data/pairs/policy_tmp"; ep_dir = tmp / "_episodes"; ep_dir.mkdir(parents=True, exist_ok=True)
scenes = {}
for f in sorted((ROOT / a.rollouts).glob("ep_*.npz")):
    task, ep = map(int, f.stem.split("_")[1:])
    key = f"{task}_{ep}"
    if (ep_dir / f"{key}.json").exists():
        continue
    z = np.load(f, allow_pickle=False)
    if task not in scenes:
        scenes[task] = Scene(task)
    sc = scenes[task]
    demo = Demo(sc, None, ep, xml=str(z["xml"]), states=z["states"], actions=z["actions"], grip=z["grip"])
    masks = render.Masks(sc)
    q = Quota(s1_pos=a.per_class, s1_neg=a.per_class, g_each=0, h_counts=(0, 0, 0, 0), max_gray=4, max_draws=40)
    arrays, metas, stats = sample_episode(sc, masks, demo, task, keep_gray=True, quota=q, k=a.k, s1_source="S3")
    if not arrays:
        print(f"[{key}] no pairs"); continue
    for m in metas:
        m.update(split="test_policy", success=bool(z["success"]), ep_len=int(len(z["actions"])))
    np.savez_compressed(ep_dir / f"{key}.npz", **{k: np.stack([x[k] for x in arrays]) for k in arrays[0]})
    (ep_dir / f"{key}.json").write_text(json.dumps({"meta": metas, "stats": stats}))
    print(f"[{key}] {len(metas)} pairs (success={bool(z['success'])})", flush=True)
