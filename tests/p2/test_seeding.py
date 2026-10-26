import numpy as np
import pytest

from second_look.seeding import REROLL, noise_seed, p5_uniform

BASE = dict(split="dev", task=3, init=7, seed_id="p2dev-1", intervention_idx=0, decision=12, k=0)


def test_stable_and_in_range():
    a = noise_seed(**BASE)
    assert a == noise_seed(**BASE) and 0 <= a < 1 << 63
    assert a == noise_seed("dev", np.int64(3), np.int32(7), "p2dev-1", np.int64(0), np.int64(12), np.int64(0))          # numpy ints hash like ints


@pytest.mark.parametrize("field,other", [("split", "test"), ("task", 4), ("init", 8), ("seed_id", "p2dev-2"), ("intervention_idx", 1), ("decision", 13), ("k", 1),
                                         ("intervention_idx", REROLL)])
def test_every_field_changes_the_seed(field, other):
    assert noise_seed(**{**BASE, field: other}) != noise_seed(**BASE)


def test_no_arm_in_the_key_so_arms_share_noise():
    """The signature has no arm argument: baseline and any arm get the same noise until their first intervention (SPEC §2.4)."""
    import inspect
    assert "arm" not in " ".join(inspect.signature(noise_seed).parameters)


def test_distinct_over_a_grid():
    seeds = {noise_seed("dev", t, i, "s", iv, d, k) for t in range(3) for i in range(3) for iv in (0, 1, REROLL) for d in range(30) for k in range(3)}
    assert len(seeds) == 3 * 3 * 3 * 30 * 3


def test_bad_intervention_rejected():
    for bad in (-1, "x", 1.5, True):
        with pytest.raises(ValueError):
            noise_seed(**{**BASE, "intervention_idx": bad})


def test_p5_uniform():
    u = p5_uniform("dev", 3, 7, "s", 0, 12)
    assert 0 <= u < 1 and u == p5_uniform("dev", 3, 7, "s", 0, 12)
    assert p5_uniform("dev", 3, 7, "s", 1, 12) != u and p5_uniform("dev", 3, 7, "s", 0, 13) != u
    draws = np.array([p5_uniform("dev", 0, i, "s", 0, d) for i in range(100) for d in range(50)])
    assert abs(draws.mean() - 0.5) < 0.02 and abs((draws < 0.1).mean() - 0.1) < 0.02          # uniform enough to be a Bernoulli source


def test_p5_key_differs_from_the_noise_key():
    assert int(p5_uniform("dev", 3, 7, "p2dev-1", 0, 12) * (1 << 63)) != noise_seed("dev", 3, 7, "p2dev-1", 0, 12, 0)


def test_fm_noise_cpu_reproducible():
    from second_look.seeding import decision_noise
    a = decision_noise("dev", 0, 0, "s", 0, 5, ks=(0, 1), device="cpu")
    b = decision_noise("dev", 0, 0, "s", 0, 5, ks=(1, 0), device="cpu")
    assert a.shape == (2, 50, 32) and (a[0] == b[1]).all() and (a[1] == b[0]).all()          # independent of batch composition
    assert not (a[0] == a[1]).all()
