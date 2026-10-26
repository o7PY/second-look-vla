"""The budget clock of one episode (SPEC_part2.md §2.1). Restoring the arm never restores the clock; LeRobot's own step counter is not used.

    elapsed = executed steps + pause steps + return-cost steps          episode over at success or when elapsed >= limit

Success is checked after EVERY physics step, including pause steps. A retry must not get free time: the steps between the rewind target and the alarm are
charged again as the "return cost" (as long as the motion took), unless the teleport sensitivity run sets the cost to 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

LIMIT = 300
PAUSE_STEPS = 10


@dataclass
class Budget:
    limit: int = LIMIT
    executed: int = 0
    paused: int = 0
    returned: int = 0
    rewinds: int = 0
    probed: int = 0                  # Part 3 probe arms: control steps spent moving between probe poses + the planning time of probe plans (zero for every Part 2 / TEST arm)

    @property
    def elapsed(self) -> int:
        return self.executed + self.paused + self.returned + self.probed

    @property
    def left(self) -> int:
        return max(0, self.limit - self.elapsed)

    @property
    def exhausted(self) -> bool:
        return self.elapsed >= self.limit

    def charge_step(self) -> None:
        """One executed control step. Never called when the budget is exhausted."""
        if self.exhausted:
            raise RuntimeError("budget exhausted")
        self.executed += 1

    def charge_pause_step(self) -> None:
        if self.exhausted:
            raise RuntimeError("budget exhausted")
        self.paused += 1

    def charge_probe_step(self) -> None:
        """One control step of a probe move (a move is real robot time). Never called when the budget is exhausted."""
        if self.exhausted:
            raise RuntimeError("budget exhausted")
        self.probed += 1

    def charge_probe_plan(self, steps: int) -> None:
        """Planning time of a probe plan that is NOT executed, charged in full (like a return cost) so that the budget check sees it."""
        self.probed += int(steps)

    def charge_return(self, steps: int) -> None:
        """A rewind: `steps` = (alarm step - target step), charged in full even if that exhausts the budget (the retry is then cut short)."""
        if steps < 0:
            raise ValueError("return cost must be >= 0")
        self.returned += int(steps)
        self.rewinds += 1


def run_pause(budget: Budget, step_fn: Callable[[], bool], k: int = PAUSE_STEPS) -> tuple[int, bool]:
    """Up to `k` pause steps. `step_fn()` makes one physics step and returns whether the task is now successful (checked after every step).
    Stops early on success or when the budget runs out. Returns (steps taken, success)."""
    n = 0
    while n < k and not budget.exhausted:
        budget.charge_pause_step()
        n += 1
        if step_fn():
            return n, True
    return n, False
