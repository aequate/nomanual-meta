"""Fixtures for the firmware host tests: one compiled nm_host per session."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from nomanual.rig.host_engine import HostBuildError, HostEngine, HostProcess, build_host_binary


@pytest.fixture(scope="session")
def host_exe():
    try:
        return build_host_binary()
    except HostBuildError as exc:  # no compiler on this machine: skip, and say so
        pytest.skip(f"firmware host build unavailable: {exc}")


@pytest.fixture
def host(host_exe) -> Iterator[HostProcess]:
    with HostProcess(host_exe) as proc:
        yield proc


@pytest.fixture
def engine(host: HostProcess) -> HostEngine:
    return HostEngine(host)
