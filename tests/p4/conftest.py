import os

import pytest


def _mem_available_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 0


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs a live LIBERO env (and RAM); run with P2_LIVE=1 and >= 6 GB MemAvailable")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("P2_LIVE") == "1" and _mem_available_mb() >= 6000:
        return
    skip = pytest.mark.skip(reason="live test: set P2_LIVE=1 and free >= 6 GB RAM (close Firefox / the proxy)")
    for it in items:
        if "live" in it.keywords:
            it.add_marker(skip)
