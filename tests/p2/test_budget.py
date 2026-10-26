import pytest

from second_look.budget import Budget, run_pause


def test_elapsed_is_the_sum():
    b = Budget()
    for _ in range(40):
        b.charge_step()
    n, ok = run_pause(b, lambda: False)
    assert (n, ok) == (10, False)
    b.charge_return(20)
    assert (b.executed, b.paused, b.returned, b.rewinds, b.elapsed) == (40, 10, 20, 1, 70)


def test_two_rewinds_scripted():
    """The G5 arithmetic without a simulator: 2 rewinds of 2 decisions each."""
    b = Budget()
    for _ in range(100):
        b.charge_step()
    run_pause(b, lambda: False); b.charge_return(20)
    for _ in range(50):
        b.charge_step()
    run_pause(b, lambda: False); b.charge_return(30)
    assert b.elapsed == (100 + 50) + 2 * 10 + (20 + 30) and b.rewinds == 2 and b.left == 300 - 220


def test_success_during_pause_counts_and_stops_the_pause():
    b = Budget()
    calls = []
    def step():
        calls.append(1)
        return len(calls) == 4
    n, ok = run_pause(b, step)
    assert (n, ok, b.paused, b.elapsed) == (4, True, 4, 4)


def test_pause_cut_short_by_the_budget():
    b = Budget(limit=300, executed=295)
    n, ok = run_pause(b, lambda: False)
    assert (n, ok, b.elapsed, b.exhausted) == (5, False, 300, True)
    with pytest.raises(RuntimeError):
        b.charge_step()


def test_return_cost_can_exhaust_the_budget_and_never_restores_the_clock():
    b = Budget(executed=250, paused=10)
    b.charge_return(60)
    assert b.exhausted and b.elapsed == 320 and b.left == 0
    b2 = Budget(); b2.charge_return(0)
    assert b2.elapsed == 0 and b2.rewinds == 1                              # teleport sensitivity run: cost 0
    with pytest.raises(ValueError):
        b2.charge_return(-1)
