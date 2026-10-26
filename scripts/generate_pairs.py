"""Generate S1 + S2 pairs. Per-episode files go to <out>/_episodes/, then `--consolidate` builds shards.

  python scripts/generate_pairs.py --config configs/data_goal.yaml                 # generate (resumable)
  python scripts/generate_pairs.py --config configs/data_goal.yaml --limit 6 --tag trial   # P3 trial run
  python scripts/generate_pairs.py --config configs/data_goal.yaml --consolidate
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from second_look.scene import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=0, help="only this many episodes per task (trial run)")
ap.add_argument("--tag", default="", help="subfolder tag for trial runs (keeps the real dataset untouched)")
ap.add_argument("--consolidate", action="store_true")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
OUT = ROOT / cfg["out_dir"] / (args.tag if args.tag else "")
EP = OUT / "_episodes"


def _free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def work(job):
    """job = (task_idx, [(demo_idx, split), ...]) -> list of (key, n_pairs, stats)"""
    import h5py
    from second_look import render
    from second_look.sampler import Demo, Quota, sample_episode
    from second_look.scene import DEMO_DIR, Scene, task_names
    task, eps = job
    name = task_names()[task]
    sc = Scene(task)
    masks = None
    done = []
    with h5py.File(DEMO_DIR / f"{name}_demo.hdf5", "r") as h:
        for demo_idx, split in eps:
            key = f"{task}_{demo_idx}"
            if (EP / f"{key}.json").exists():
                continue
            if _free_gb() < cfg["min_free_gb"]:
                print(f"[stop] free disk < {cfg['min_free_gb']} GB", file=sys.stderr)
                return done
            demo = Demo(sc, h, demo_idx)
            masks = render.Masks(sc)            # registry is rebuilt per demo model
            q = cfg["quota"]
            quota = Quota(q["s1_pos"], q["s1_neg"], q["g_each"], tuple(q["h_counts"]), q["max_gray"], q["max_draws"])
            t0 = time.time()
            arrays, metas, stats = sample_episode(sc, masks, demo, task, keep_gray=split.startswith("test"),
                                                  quota=quota, k=cfg["k"])
            for m in metas:
                m["split"] = split
            np.savez_compressed(EP / f"{key}.npz", **{k: np.stack([a[k] for a in arrays]) for k in arrays[0]})
            (EP / f"{key}.json").write_text(json.dumps({"meta": metas, "stats": stats, "secs": time.time() - t0}))
            done.append((key, len(metas), time.time() - t0))
            print(f"[{key}] {split:18s} {len(metas):4d} pairs  {time.time()-t0:5.1f}s", flush=True)
    return done


def generate():
    from second_look.sampler import load_or_make_splits, splits_sha256
    EP.mkdir(parents=True, exist_ok=True)
    sp = load_or_make_splits()
    print("splits.json sha256:", splits_sha256())
    by_task: dict[int, list] = {}
    for key, split in sp["episodes"].items():
        t, d = map(int, key.split(":"))
        by_task.setdefault(t, []).append((d, split))
    jobs = []
    for t, eps in sorted(by_task.items()):
        eps = sorted(eps)[: args.limit] if args.limit else sorted(eps)
        for i in range(0, len(eps), 5):
            jobs.append((t, eps[i:i + 5]))
    print(f"{len(jobs)} jobs, {sum(len(j[1]) for j in jobs)} episodes, {cfg['workers']} workers, out={OUT}")
    t0 = time.time()
    with mp.get_context("spawn").Pool(cfg["workers"]) as pool:
        for res in pool.imap_unordered(work, jobs):
            pass
    print(f"generation done in {(time.time()-t0)/60:.1f} min; free disk {_free_gb():.1f} GB")


def consolidate():
    import pandas as pd
    files = sorted(EP.glob("*.json"))
    per_split: dict[str, list] = {}
    for f in files:
        j = json.loads(f.read_text())
        if not j["meta"]:
            continue
        per_split.setdefault(j["meta"][0]["split"], []).append(f.stem)
    for split, keys in per_split.items():
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        buf, rows, shard = {}, [], 0
        def flush():
            nonlocal buf, shard
            if not buf:
                return
            n = len(next(iter(buf.values())))
            np.savez_compressed(d / f"shard-{shard:04d}.npz", **{k: np.concatenate(v) for k, v in buf.items()})
            print(f"  {split}/shard-{shard:04d}.npz  {n} pairs", flush=True)
            shard += 1
            buf = {}
        for key in keys:
            z = np.load(EP / f"{key}.npz")
            meta = json.loads((EP / f"{key}.json").read_text())["meta"]
            arrs = {k: z[k] for k in z.files}
            for i in range(len(meta)):
                for k in arrs:
                    buf.setdefault(k, []).append(arrs[k][i:i + 1].copy())    # copy: a slice would pin the whole episode array
                m = meta[i]
                m["shard"] = shard
                m["row"] = len(buf["ref"]) - 1
                rows.append(m)
                if len(buf["ref"]) == cfg["shard_size"]:
                    flush()
        flush()
        pd.DataFrame(rows).to_parquet(d / "meta.parquet")
        print(f"{split}: {len(rows)} pairs, {len(keys)} episodes")


def main():
    if args.consolidate:
        consolidate()
    else:
        generate()


if __name__ == "__main__":
    main()
