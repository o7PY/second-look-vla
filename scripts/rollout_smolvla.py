"""S3 pass 1: SmolVLA rollouts (20 episodes x 10 tasks), exact per-step state recording. Resumable.
  python scripts/rollout_smolvla.py --episodes 20            # all tasks
  python scripts/rollout_smolvla.py --episodes 1 --tasks 0 --out results/part1/policy_rollouts_smoke
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from second_look.scene import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
ap.add_argument("--out", default="results/part1/policy_rollouts"); ap.add_argument("--n_action_steps", type=int, default=10)
a = ap.parse_args()
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

from second_look import policy_rollout as PR
PR.patch_renderer()
pol = PR.Policy(n_action_steps=a.n_action_steps)
summary = {}
for task in a.tasks:
    env = PR.make_env(task, 0)
    for ep in range(a.episodes):
        f = out / f"ep_{task}_{ep}.npz"
        if f.exists():
            continue
        t0 = time.time()
        r = PR.rollout(env, pol, seed=1000 * task + ep)
        np.savez_compressed(f, states=r["states"], actions=r["actions"], grip=r["grip"], xml=np.array(r["xml"]), success=r["success"])
        print(f"task {task} ep {ep}: {len(r['actions'])} steps, success={r['success']}, {time.time()-t0:.0f}s", flush=True)
    env.close()
for task in a.tasks:
    fs = sorted(out.glob(f"ep_{task}_*.npz"))
    summary[task] = {"episodes": len(fs), "success": float(np.mean([bool(np.load(f)["success"]) for f in fs])) if fs else None}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(summary)
