"""R5 pass 1: SmolVLA rollouts for the SEALED set test_policy2. 20 episodes x 10 tasks. Resumable. Prints only counts and success rates.
Differences from scripts/rollout_smolvla.py (Part 1):
  - seed namespace "policy2": episode seed = sha256("policy2|task|ep") (never 1000*task+ep as in Part 1); asserted disjoint from the Part 1 seeds;
    the policy's flow-matching noise is seeded per episode from the same hash (Part 1 did not seed it);
  - init states 20..39 of the 50 LIBERO init states of every task (Part 1 used 0..19), so test_policy and test_policy2 share no initial condition;
  - the fixed restore (R3) is irrelevant here (live rollouts), it matters in pass 2.
  python scripts/rollout_policy2.py [--episodes 20] [--tasks ...]"""
import argparse, hashlib, json, time
import numpy as np
from second_look.scene import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--tasks", type=int, nargs="*", default=list(range(10)))
ap.add_argument("--out", default="results/part1/policy_rollouts2"); ap.add_argument("--n_action_steps", type=int, default=10)
ap.add_argument("--init_start", type=int, default=20)
a = ap.parse_args()
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

def seed_for(task, ep): return int.from_bytes(hashlib.sha256(f"policy2|{task}|{ep}".encode()).digest()[:4], "big") & 0x7FFFFFFF
part1 = {1000 * t + e for t in range(10) for e in range(20)}
seeds = {(t, e): seed_for(t, e) for t in range(10) for e in range(a.episodes)}
assert not (set(seeds.values()) & part1), "seed overlap with test_policy"
assert len(set(seeds.values())) == len(seeds), "duplicate policy2 seeds"
assert a.init_start >= 20 and a.init_start + a.episodes <= 50, "init states must be unused by test_policy (0..19) and exist (< 50)"

import torch
from second_look import policy_rollout as PR
PR.patch_renderer()
pol = PR.Policy(n_action_steps=a.n_action_steps)
manifest = {}
for task in a.tasks:
    env = PR.make_env(task, 0)
    for ep in range(a.episodes):
        f = out / f"ep_{task}_{ep}.npz"
        if f.exists():
            continue
        # LiberoEnv increments init_state_id at every reset, and also auto-resets on success, so the counter drifts: set it explicitly per episode
        sid = a.init_start + ep
        env.init_state_id = sid
        torch.manual_seed(seeds[(task, ep)]); np.random.seed(seeds[(task, ep)])
        t0 = time.time()
        r = PR.rollout(env, pol, seed=seeds[(task, ep)])
        np.savez_compressed(f, states=r["states"], actions=r["actions"], grip=r["grip"], xml=np.array(r["xml"]), success=r["success"],
                            seed=seeds[(task, ep)], init_state_id=sid)
        print(f"task {task} ep {ep}: init {sid}, {len(r['actions'])} steps, {time.time()-t0:.0f}s", flush=True)   # no success flag per episode in the log
    env.close()
rows = {}
for task in a.tasks:
    fs = sorted(out.glob(f"ep_{task}_*.npz"))
    zs = [np.load(f) for f in fs]
    rows[task] = {"episodes": len(fs), "success_rate": float(np.mean([bool(z["success"]) for z in zs])) if zs else None, "init_state_ids": sorted(int(z["init_state_id"]) for z in zs)}
json.dump({"namespace": "policy2", "init_start": a.init_start, "seeds": {f"{t}_{e}": s for (t, e), s in seeds.items()}, "tasks": rows}, open(out / "manifest.json", "w"), indent=1)
print({t: (r["episodes"], r["success_rate"]) for t, r in rows.items()})
