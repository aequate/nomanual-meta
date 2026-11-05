"""SIGTERM to the bench runner stops its worker processes too (Linux: reads /proc)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

RUNNER = """
import sys
from pathlib import Path
from nomanual.core import bench
from nomanual.core.benchrun import run_set
out = Path(sys.argv[1])
run_set(bench.load("dev")[:2], ["enum"], out, budget=80, workers=2)
"""


def _children(pid: int) -> list[int]:
    kids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text()
            except OSError:
                continue
            if int(stat.rsplit(")", 1)[1].split()[1]) == pid:
                kids.append(int(d.name))
    return kids


def _alive(pid: int) -> bool:
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_sigterm_to_the_runner_stops_its_workers(tmp_path: Path) -> None:
    proc = subprocess.Popen([sys.executable, "-c", RUNNER, str(tmp_path / "out.jsonl")])
    workers: list[int] = []
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and len(workers) < 2:  # the pool's two workers (+ tracker)
            time.sleep(0.5)
            workers = [p for p in _children(proc.pid)
                       if "spawn_main" in (Path("/proc") / str(p) / "cmdline").read_text(errors="replace")]
        assert len(workers) >= 2, "the pool did not start"
        time.sleep(2.0)  # both are inside a run now
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 128 + signal.SIGTERM
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(_alive(p) for p in workers):
            time.sleep(0.2)
        assert not any(_alive(p) for p in workers), "a worker outlived its runner"
    finally:
        if proc.poll() is None:
            proc.kill()
        for p in workers:  # never leave an orphan behind, even when the test fails
            if _alive(p):
                os.kill(p, signal.SIGKILL)
