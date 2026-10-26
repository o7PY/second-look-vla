"""Shared helpers of the live Part 2 tests: one Scene at a time (always closed), loaded with a demo episode."""
import contextlib
import gc

import h5py
import numpy as np

from second_look.scene import DEMO_DIR, Scene, task_names

NAMES = task_names()


@contextlib.contextmanager
def scene_for(task_name: str, demo: int = 0):
    """Yields (scene, states, actions). One Scene alive at a time: the first costs ~1.5 GB of RAM (see tests/test_b2_render_state.py)."""
    sc = Scene(NAMES.index(task_name))
    try:
        with h5py.File(DEMO_DIR / f"{task_name}_demo.hdf5", "r") as h:
            g = h[f"data/demo_{demo}"]
            sc.load_demo_model(g.attrs["model_file"])
            yield sc, g["states"][:], g["actions"][:]
    finally:
        sc.env.close()
        del sc
        gc.collect()


def object_xyz(scene) -> dict:
    return {k: v[:3].copy() for k, v in scene.read().items() if len(v) == 7}
