"""Benchmark sets: composition, hash check, the held-out lock."""

from __future__ import annotations

import shutil
import os
from pathlib import Path

import pytest

from nomanual.core import bench


def test_composition_meets_the_spec() -> None:
    for name, n in (("dev", 20), ("heldout", 30)):
        rows = bench.composition(name)  # type: ignore[arg-type]
        assert len(rows) == n
        assert sum(t for _, t, _ in rows) >= n / 2 and sum(s for _, _, s in rows) >= n / 3
    assert not set(bench.SEEDS["dev"]) & set(bench.SEEDS["heldout"])


def test_dev_loads_and_matches_its_recorded_hash() -> None:
    items = bench.load("dev")
    assert [it.index for it in items] == list(range(20)) and items[0].name == "dev-00"
    assert {(it.seed, it.trap, it.swap) for it in items} == set(bench.composition("dev"))
    assert all(it.config.thresholds for it in items if it.trap)


def test_heldout_is_locked_without_the_flag() -> None:
    with pytest.raises(bench.HeldOutLocked):
        bench.load("heldout")


def test_a_changed_set_file_is_refused(tmp_path: Path) -> None:
    shutil.copy(bench.BENCH_DIR / "README.md", tmp_path / "README.md")
    data = (bench.BENCH_DIR / "dev.json").read_bytes().replace(b"\r\n", b"\n")  # as committed (LF)
    (tmp_path / "dev.json").write_bytes(data.replace(b'"seed":1000', b'"seed":1001', 1))
    with pytest.raises(bench.HashMismatch):
        bench.load("dev", tmp_path)
    (tmp_path / "dev.json").write_bytes(data.replace(b"\n", b"\r\n"))  # a CRLF checkout is the same file
    assert len(bench.load("dev", tmp_path)) == 20


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_redraw_matches_the_stored_sets() -> None:
    stored = (bench.BENCH_DIR / "dev.json").read_bytes()
    assert bench.sha256(bench.dumps(bench.draw("dev"))) == bench.sha256(stored)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(os.environ.get("NOMANUAL_UNLOCK_HELDOUT_TESTS") != "1",
                    reason="held-out fixture access requires explicit human opt-in")
def test_heldout_redraw_requires_human_opt_in() -> None:
    stored = (bench.BENCH_DIR / "heldout.json").read_bytes()
    assert bench.sha256(bench.dumps(bench.draw("heldout"))) == bench.sha256(stored)


def test_fresh_machines_follow_the_composition_rule_and_refuse_set_seeds() -> None:
    items = bench.fresh([7000, 7001, 7002, 7003])
    assert [it.name for it in items] == ["fresh-00", "fresh-01", "fresh-02", "fresh-03"]
    assert [(it.trap, it.swap) for it in items] == [(True, True), (False, False), (True, False), (False, True)]
    assert items[0].seed == 7000 and items[0].config == bench.fresh([7000])[0].config  # deterministic
    for taken in (1003, 5007):  # a dev seed, a held-out seed
        with pytest.raises(ValueError, match="dev or held-out"):
            bench.fresh([7000, taken])
