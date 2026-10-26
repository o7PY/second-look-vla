import math

import numpy as np

from second_look.cusum import CUSUM, CusumParams, calibrate, conformal_height, first_alarm, peaks


def test_recursion_by_hand():
    p = CusumParams.make(mu0=1.0, sigma=2.0, h=1.4)          # k = 0.5
    c = CUSUM(p)
    out = [c.update(x) for x in (1.0, 2.0, 3.0, 0.0)]
    assert [round(s, 9) for s, _ in out] == [0.0, 0.5, 2.0, 0.5]          # max(0, .) then +0.5, +1.5, then 2.0 - 1.5
    assert [a for _, a in out] == [False, False, True, False]


def test_state_restore_is_exact():
    p = CusumParams.make(0, 1, 5)
    c = CUSUM(p)
    for x in (1, 2, 3):
        c.update(x)
    s = c.state
    c.update(10.0)
    c.restore(s)
    assert c.state == s


def test_conformal_rank():
    pk = np.arange(1.0, 10.0)                       # M = 9, alpha = 0.1: ceil(10 * 0.9) = 9 -> 9th smallest
    assert conformal_height(pk, 0.1) == 9.0
    assert conformal_height(np.arange(1.0, 6.0), 0.1) == math.inf       # M = 5: rank 6 > 5


def test_calibrate_controls_false_alarms():
    """Split-conformal coverage holds on average over calibration draws (one draw of 99 episodes has sd ~0.03 around 0.10)."""
    fpr = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        p = calibrate([rng.normal(0, 1, 30) for _ in range(99)], 0.1)
        fpr.append(np.mean([first_alarm(rng.normal(0, 1, 30), p) is not None for _ in range(500)]))
        assert first_alarm(np.full(30, 6.0), p) is not None
    assert 0.07 <= np.mean(fpr) <= 0.13, np.mean(fpr)
