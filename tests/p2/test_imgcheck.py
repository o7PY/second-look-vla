from pathlib import Path

import numpy as np
import pytest

from second_look.imgcheck import DegenerateImage, check_image, check_obs, measures, problems

FIX = np.load(Path(__file__).parent / "fixtures" / "camera_images.npz")


def test_good_images_pass():
    check_image(FIX["good_agent"], "fixture")
    check_image(FIX["good_wrist"], "fixture", agentview=False)
    check_obs({"pixels": {"image": FIX["good_agent"], "image2": FIX["good_wrist"]}}, "fixture")


def test_corrupted_render_of_the_bug_is_caught():
    for k, av in (("bad_agent", True), ("bad_wrist", False)):
        with pytest.raises(DegenerateImage, match="striped"):
            check_image(FIX[k], "fixture", agentview=av)
    with pytest.raises(DegenerateImage):
        check_obs({"pixels": {"image": FIX["bad_agent"], "image2": FIX["good_wrist"]}}, "fixture")


def test_synthetic_failures():
    rng = np.random.default_rng(0)
    assert problems(np.full((360, 360, 3), 120, np.uint8))                                     # blank
    tile = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    assert any("repeats" in p for p in problems(np.tile(tile, (6, 6, 1))))                      # tiled
    stripes = np.zeros((360, 360, 3), np.uint8); stripes[:, ::2] = 255
    assert any("striped" in p for p in problems(stripes))                                       # vertical hatching


def test_real_renders_are_not_flagged():
    """20 stored good renders from the Part 1 test_policy split (skipped if the data are not there)."""
    p = Path(__file__).resolve().parents[2] / "data/pairs/test_policy/shard-0000.npz"
    if not p.exists():
        pytest.skip("no data")
    z = np.load(p)
    for i in range(0, 600, 30):
        for key in ("ref", "set"):
            assert not problems(z[key][i]), (i, key, measures(z[key][i]))
