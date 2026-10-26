"""C3: a pair replayed from its Spec reproduces label_v1 exactly (class, gray flag, changed items); labels_v2 is consistent with v1. Needs the simulator."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from second_look.scene import ROOT

SPEC = importlib.util.spec_from_file_location("c3_labels_v2", ROOT / "scripts/c3_labels_v2.py")


@pytest.fixture(scope="module")
def c3():
    old = sys.argv
    sys.argv = ["x"]
    try:
        mod = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def test_replay_reproduces_v1_labels(c3):
    m = pd.read_parquet(ROOT / "data/pairs/val/meta.parquet")
    m["uid"] = m.pair_id + "#" + m.groupby("pair_id").cumcount().astype(str)
    rng = np.random.default_rng(0)
    pick = []
    for gen in ("S1", "G1", "G3", "G4", "H1", "H3"):                        # a spread of sources / disturbances
        pick += list(rng.choice(np.flatnonzero((m.gen == gen).values), 2, replace=False))
    for task, g in m.iloc[pick].groupby("task"):
        out = c3.replay_task(("val", int(task), g.to_dict("records")))
        for r in out:
            row = m[m.uid == r["pair_id"]].iloc[0]
            assert r["ok"]
            assert r["y1"] == int(row.label) and r["gray1"] == bool(row.gray)
            assert r["items1"] == (row.changed_items if isinstance(row.changed_items, str) else "")
            assert abs(r["trans1"] - row.max_trans_cm) < 0.1 and abs(r["rot1"] - row.max_rot_deg) < 2.0   # contact-sensitive pairs differ slightly (C3 notes)


@pytest.mark.parametrize("split", ["train", "val", "test", "test_heldout_tasks", "test_policy"])
def test_labels_v2_file_is_consistent_with_v1(split):
    m = pd.read_parquet(ROOT / "data/pairs" / split / "meta.parquet")
    l = pd.read_parquet(ROOT / "data/pairs" / split / "labels_v2.parquet")
    assert len(l) == len(m) and (l.pair_id.values == m.pair_id.values).all()
    assert (l.label_v1.values == m.label.values).all() and (l.gray_v1.values == m.gray.values).all()
    changed = (l.label_v2.values != l.label_v1.values) | (l.gray_v2.values != l.gray_v1.values)
    assert (l.label_v2.values[changed] == 0).all()                           # v2 never makes a pair more changed
    assert (m.max_trans_cm.values[changed] <= 1.0).all()                     # only pairs that did not move > 1 cm can flip
