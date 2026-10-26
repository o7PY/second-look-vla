"""R5 pass 2: pairs for the SEALED set from the policy2 rollouts: S1-style (no disturbance, pause only), <= 10 changed + <= 10 unchanged per
episode, gray <= 4, label_v2 directly, fixed restore (R3) active. Prints ONLY counts. Output: data/pairs/test_policy2/ (shards + meta.parquet),
hashes in results/part1/b2/notes/r5.md.   python scripts/make_policy2_pairs.py [--rollouts results/part1/policy_rollouts2]"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
from second_look.scene import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", default="results/part1/policy_rollouts2"); ap.add_argument("--per_class", type=int, default=10); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--consolidate", action="store_true")
a = ap.parse_args()
TMP = ROOT / "data/pairs/policy2_tmp"; EP = TMP / "_episodes"

if not a.consolidate:
    from second_look import render, sampler
    from second_look.labels_v2 import label_pair_v2
    from second_look.sampler import Demo, Quota, sample_episode
    from second_look.scene import Scene
    sampler.LABELER = label_pair_v2
    EP.mkdir(parents=True, exist_ok=True)
    scene = None
    for f in sorted((ROOT / a.rollouts).glob("ep_*.npz")):
        task, ep = map(int, f.stem.split("_")[1:]); key = f"{task}_{ep}"
        if (EP / f"{key}.json").exists():
            continue
        z = np.load(f, allow_pickle=False)
        if scene is not None and scene.task_idx != task:        # one Scene at a time (RAM)
            scene.env.close(); scene = None
        if scene is None:
            scene = Scene(task)
        demo = Demo(scene, None, ep, xml=str(z["xml"]), states=z["states"], actions=z["actions"], grip=z["grip"])
        masks = render.Masks(scene)
        q = Quota(s1_pos=a.per_class, s1_neg=a.per_class, g_each=0, h_counts=(0, 0, 0, 0), max_gray=4, max_draws=40)
        arrays, metas, stats = sample_episode(scene, masks, demo, task, keep_gray=True, quota=q, k=a.k, s1_source="S3b")
        for m in metas:
            m.update(split="test_policy2", success=bool(z["success"]), ep_len=int(len(z["actions"])), init_state_id=int(z["init_state_id"]), rollout_seed=int(z["seed"]))
        if arrays:
            np.savez_compressed(EP / f"{key}.npz", **{k: np.stack([x[k] for x in arrays]) for k in arrays[0]})
        (EP / f"{key}.json").write_text(json.dumps({"meta": metas, "stats": stats}))
        print(f"episode {key}: {len(metas)} pairs", flush=True)             # counts only
    if scene is not None: scene.env.close()
else:
    # consolidate through the same code as the other splits (writes data/pairs/policy2_tmp/test_policy2/shard-*.npz + meta.parquet)
    cfg = ROOT / "configs/data_goal.yaml"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/generate_pairs.py"), "--config", str(cfg), "--tag", "policy2_tmp", "--consolidate"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit(r.returncode)
    dst = ROOT / "data/pairs/test_policy2"
    if dst.exists(): import shutil; shutil.rmtree(dst)
    (TMP / "test_policy2").rename(dst)
    import pandas as pd
    m = pd.read_parquet(dst / "meta.parquet"); ng = m[~m.gray]
    h = hashlib.sha256()
    for f in sorted(dst.glob("*")):
        h.update(f.name.encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    sha = h.hexdigest()
    n_ep = m.groupby(["task", "demo"]).ngroups
    succ = m.groupby(["task", "demo"]).success.first().groupby(level=0).mean()
    lines = ["# R5: sealed policy test set `test_policy2`", "",
             "Built by `scripts/rollout_policy2.py` (SmolVLA, 20 episodes x 10 tasks, `n_action_steps` 10, new seed namespace `policy2`, LIBERO init states 20-39 of 50 per task, none used by `test_policy`) and `scripts/make_policy2_pairs.py` (S1-style pairs, <= 10 changed + <= 10 unchanged per episode, gray <= 4, `label_v2` directly, the R3 restore fix active, source tag `S3b`).",
             "", "**Sealed:** no model has been run on it and no image or per-pair statistic has been looked at; this note lists counts and success rates only. `src/second_look/sealed.py` makes `PairDataset` / `build_cache` refuse `test_policy2` until `results/part1/b2/notes/r6_decision.md` is committed (`tests/test_b2_sealed.py`).", "",
             f"- episodes: {n_ep}; pairs: {len(m)} (unchanged {int((ng.label == 0).sum())}, changed {int((ng.label == 1).sum())}, gray {int(m.gray.sum())})",
             "- task success rates of the rollouts: " + ", ".join(f"task {int(t)}: {v:.2f}" for t, v in succ.items()) + f" (overall {float(m.groupby(['task','demo']).success.first().mean()):.3f})",
             f"- directory hash (sha256 over file names and file hashes of `data/pairs/test_policy2/`): `{sha}`", "",
             "## Overlap with `test_policy`", "- rollout seeds: disjoint (asserted: policy2 seeds are sha256-derived, Part 1 used 1000*task+episode);",
             "- initial states: `test_policy` used LIBERO init states 0-19 (the environment counter starts at 0 and increments per episode); `test_policy2` uses 20-39 (set explicitly per episode because LIBERO also auto-resets on success). No init state is shared. The scene layout comes from the same 10 tasks, so the *tasks* are the same as in `test_policy`, with new initial object placements and new policy runs."]
    Path(ROOT / "results/part1/b2/notes/r5.md").write_text("\n".join(lines) + "\n")
    print(f"test_policy2: {n_ep} episodes, {len(m)} pairs (unchanged {int((ng.label == 0).sum())}, changed {int((ng.label == 1).sum())}, gray {int(m.gray.sum())}); sha256 {sha[:16]}…")
