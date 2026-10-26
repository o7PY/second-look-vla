"""G6 inputs: fresh denoise paths and STAC resamples recorded from this project's own SmolVLA rollouts (the old pilot runs do not exist, see results/part2/notes/deviations.md).
10 tasks x 3 episodes (LIBERO inits 40, 41, 42 = CAL range, baseline only, up to 30 decisions each): per decision the (11, 50, 7) iterates of the executed chunk and the 8 batched
resamples (8, 50, 7) -> results/part2/g6_inputs.npz, plus each episode's baseline success.
  scripts/safe_run.sh python scripts/p2_g6_record.py"""
import time
import numpy as np

import second_look.loop as L
from second_look.cusum import CusumParams
from second_look.scene import ROOT

OUT = ROOT / "results" / "part2" / "g6_inputs.npz"
if __name__ == "__main__":
    rig = L.Rig()
    its, smp, ep, dd, succ = [], [], [], [], []
    t0 = time.time()
    for task in range(10):
        for init in (40, 41, 42):
            cfg = L.RunCfg("g6", task, init, "g6", CusumParams.make(0.0, 1.0, 1e12), np.ones(70, np.float32), stac=CusumParams.make(0.0, 1.0, 1e12), keep_raw=True)
            r = L.run_episode(rig, cfg, [])["P0"]
            for d, (i, s) in enumerate(r["raw"]):
                its.append(i), smp.append(s), ep.append(task * 100 + init), dd.append(d)
            succ.append(bool(r["success"]))
            print(f"task {task} init {init}: {len(r['raw'])} decisions, success={r['success']}, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, iterates=np.stack(its), samples=np.stack(smp), episode=np.array(ep), decision=np.array(dd), success=np.array(succ))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(its)} decisions")
    rig.close()
