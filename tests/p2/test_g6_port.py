"""G6: the `src/second_look/` accel and STAC equal the reference fmaccel (third_party/fm-geometry @ 1cb8af4 + patch) on the SAME fresh inputs: np.allclose(rtol=1e-4, atol=1e-6).
Inputs: results/part2/g6_inputs.npz (scripts/p2_g6_record.py). The fmaccel imports are CPU-only numpy code. Also writes results/part2/g6_report.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from second_look import detectors as D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "fm-geometry"))
INPUTS = ROOT / "results" / "part2" / "g6_inputs.npz"
pytestmark = pytest.mark.skipif(not INPUTS.exists(), reason="run scripts/p2_g6_record.py first")


@pytest.fixture(scope="module")
def data():
    z = np.load(INPUTS)
    return {k: z[k] for k in z.files}


def test_accel_equals_reference(data):
    from fmaccel.geometry.accel import score_chunks
    it = data["iterates"]                                                      # (N, 11, 50, 7)
    scale = D.fit_accel_scale([D.exec_window(x) for x in it])
    mine = np.array([D.accel(x, scale) for x in it])
    ref_fixed = score_chunks(it, 7, n_exec=10, mode="accel_prefix:7", fixed_std=scale)           # same scale handed over
    ref_batch = score_chunks(it, 7, n_exec=10, mode="accel_prefix:7")                           # reference's own pooled in-batch std
    assert it.shape[0] >= 250 and np.isfinite(mine).all()
    assert np.allclose(mine, ref_fixed, rtol=1e-4, atol=1e-6), np.abs(mine - ref_fixed).max()
    assert np.allclose(mine, ref_batch, rtol=1e-4, atol=1e-6), "our fitted scale differs from the reference's pooled std"
    rep = {"accel": {"n": int(len(mine)), "max_abs_diff_fixed": float(np.abs(mine - ref_fixed).max()), "max_abs_diff_batch_std": float(np.abs(mine - ref_batch).max()),
                     "min": float(mine.min()), "max": float(mine.max()), "scale_len": int(len(scale))}}
    Path(ROOT / "results/part2/g6_report.json").write_text(json.dumps(rep, indent=1))


def test_stac_equals_reference(data):
    from fmaccel.detectors.base import ChunkRecord
    from fmaccel.detectors.stac import StacDetector
    smp, ep, dd = data["samples"], data["episode"], data["decision"]           # (N, 8, 50, 7)
    det = StacDetector(n_exec=10)
    mine, ref = [], []
    import inspect
    fields = set(inspect.signature(ChunkRecord).parameters)
    for i in range(len(smp)):
        prev = smp[i - 1] if i > 0 and ep[i - 1] == ep[i] else None
        mine.append(D.stac(smp[i], prev))
        if prev is None:
            ref.append(0.0)                                                    # reference: first decision has no previous set -> 0.0
            continue
        kw = {"x_t": np.zeros((11, 50, 7), np.float32), "action_dim": 7, "n_exec": 10, "resample_full": smp[i], "prev_resample_full": prev}
        ref.append(det.score(ChunkRecord(**{k: v for k, v in kw.items() if k in fields})))
    mine, ref = np.array(mine), np.array(ref)
    assert (mine[dd == 0] == 0).all() and (mine > 0).sum() > 200
    assert np.allclose(mine, ref, rtol=1e-4, atol=1e-6), np.abs(mine - ref).max()
    p = Path(ROOT / "results/part2/g6_report.json")
    rep = json.loads(p.read_text()) if p.exists() else {}
    rep["stac"] = {"n": int(len(mine)), "max_abs_diff": float(np.abs(mine - ref).max()), "min": float(mine.min()), "max": float(mine.max())}
    p.write_text(json.dumps(rep, indent=1))
