"""Build the uncompressed memory-mapped pair cache used by training / baselines / eval (random access to the
compressed npz shards is too slow and RAM-hungry).  python scripts/cache_features.py --split all
We deliberately cache *pixels*, not frozen DINOv2 features: 60k pairs x 2 x 256 x 384 fp16 = 23 GB, and B3 then
could not use augmentation. B2/B3 run the frozen encoder on the fly instead (forward only, no_grad)."""
import argparse, time
from second_look.dataset import SPLITS, build_cache
from second_look.scene import ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="all"); ap.add_argument("--force", action="store_true")
a = ap.parse_args()
for sp in (SPLITS if a.split == "all" else [a.split]):
    if not (ROOT / "data/pairs" / sp / "meta.parquet").exists():
        print(f"skip {sp} (no pairs yet)"); continue
    t = time.time(); d = build_cache(sp, force=a.force)
    print(f"{sp}: cached in {time.time()-t:.0f}s -> {d}", flush=True)
