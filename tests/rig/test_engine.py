"""The firmware engine (firmware/src/engine.cpp, compiled on the host) against the reference.

Golden traces come from the core's exporter (`bench/machine_golden.json` on main, made by
`scripts/export_configs.py` with `nomanual/core/machine.py`). Random-config parity lives in
test_parity.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomanual.core.machine import Machine, MachineConfig, hold_ticks
from nomanual.rig.host_engine import HostEngine, HostProcess, config_fields
from tests.core.machine_strategies import make_config

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "bench" / "machine_golden.json"
EXPORT = ROOT / "bench" / "configs_export.json"
HEADER = ROOT / "firmware" / "include" / "configs.h"
TOL = 1e-6


def test_golden_traces_match(engine: HostEngine) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["cases"], "no golden cases"
    worst = 0.0
    for case in golden["cases"]:
        cfg = MachineConfig.model_validate(case["config"])
        engine.configure(cfg)
        for i, step in enumerate(case["steps"]):
            if step["swap_before"]:
                engine.swap()
            act = step["action"]
            ms = round(act["duration"] * 1000)
            n_hold = engine.hold_ticks_ms(ms)
            assert n_hold == hold_ticks(act["duration"]), (cfg.id, i)
            got = engine.run_ticks(act["channels"], n_hold)
            worst = max(worst, abs(got.u - step["u"]), abs(got.v - step["v"]))
            assert got.u == pytest.approx(step["u"], abs=TOL), (cfg.id, i)
            assert got.v == pytest.approx(step["v"], abs=TOL), (cfg.id, i)
            assert got.mode == step["mode"], (cfg.id, i)
            assert got.led == step["led"], (cfg.id, i)
    assert worst == 0.0, f"engine matches within tolerance but not bit for bit (worst {worst:g})"


def test_compiled_draw_table_matches_the_export(engine: HostEngine) -> None:
    """DRAW n (the table compiled from configs.h) behaves exactly like draw n of the JSON export."""
    export = json.loads(EXPORT.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for d, case in zip(export["draws"], golden["cases"], strict=True):
        cfg = MachineConfig.model_validate(d["config"])
        assert engine.draw(d["draw"]) == cfg.id
        ref = Machine(cfg)
        for i, step in enumerate(case["steps"]):
            if step["swap_before"]:
                engine.swap()
                ref.swap()
            act = step["action"]
            n = hold_ticks(act["duration"])
            got = engine.run_ticks(act["channels"], n)
            ref.run_ticks(act["channels"], n)
            assert (got.u, got.v, got.mode, got.led) == (ref.pos[0], ref.pos[1], ref.mode, ref.led), (cfg.id, i)


def test_header_is_the_exported_one() -> None:
    """firmware/include/configs.h is the core exporter's header (regenerate with rig_sync_configs.py)."""
    exported = (ROOT / "bench" / "configs_export.h").read_bytes().replace(b"\r\n", b"\n")
    assert HEADER.read_bytes().replace(b"\r\n", b"\n") == exported


def test_hold_ticks_from_ms_matches_reference(engine: HostEngine) -> None:
    for ms in range(200, 2001):  # every ms; the old /DT formula differed on 235, 295, 545, ...
        assert engine.hold_ticks_ms(ms) == hold_ticks(ms / 1000), ms


def test_traced_ticks_match_reference_every_tick(engine: HostEngine) -> None:
    cfg = make_config(
        conditional=[{"channel": "A", "mode": 1, "effect": {"gain_mult": 2.0}}],
        thresholds=[{"axis": "u", "op": ">", "value": 0.32, "channel": "A",
                     "effect": {"drop": {"axis": "v", "amount": 0.07}}}],
        indicator={"present": True, "shows": "mode"},
    )
    engine.configure(cfg)
    ref = Machine(cfg)
    for channels, n in [({"B": 1.0}, 20), ({"A": 1.0, "C": 0.5}, 150), ({"A": -0.7, "C": 1.0, "D": 0.3}, 200)]:
        path, end = engine.run_ticks_traced(channels, n)
        trace = ref.run_ticks(channels, n, record=True)
        assert path == trace.path
        assert (end.mode, end.led, end.drops) == (trace.mode_after, ref.led, trace.drops)


def test_rejects_invalid_configs(host: HostProcess) -> None:
    eng = HostEngine(host)
    good = make_config()
    eng.configure(good)
    fields = config_fields(good).split(" ")
    # roles are fields 2..5 (after "CFG" and the id): make two interlocks.
    bad_roles = fields.copy()
    bad_roles[2:6] = ["2", "2", "2", "3"]
    assert host.cmd(" ".join(bad_roles))[0].startswith("ERR")
    assert host.cmd("CFG x")[0].startswith("ERR")
    assert host.cmd("DRAW 0")[0] == "ERR draw"
    assert host.cmd("DRAW 99")[0] == "ERR draw"


def test_unconfigured_engine_does_not_move(host_exe) -> None:
    with HostProcess(host_exe) as proc:
        out = proc.cmd("TICK 3ff0000000000000 3ff0000000000000 3ff0000000000000 3ff0000000000000 1")
        assert out[0].startswith("POS ")
        assert HostEngine._parse_pos(out[0]).u == 0.5
