"""Export the rig's draw table for the firmware, plus golden traces for the parity test.

Writes (default `bench/`):
- `configs_export.h`  C header with the draw table (draw n = NM_CONFIGS[n-1]; draw 0 is the
                      firmware's calibration mode). Grok copies it to `firmware/include/configs.h`.
- `configs_export.json` the same configurations as CONTRACTS §3.2 JSON, plus the board.
- `machine_golden.json` for every exported config, a fixed action sequence and the reference
                      machine's state after every action (servo u, v, mode, led), with a
                      swap half way. Firmware engine output must match within 1e-6.

Usage: python scripts/export_configs.py [--board board.json] [--n 8] [--out bench]
The board defaults to the CONTRACTS §3.3 example; pass the measured physical board instead.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nomanual.core.board import Board
from nomanual.core.env import CHANNELS, Action
from nomanual.core.generator import Draw, generate
from nomanual.core.machine import Machine, MachineConfig

RIG_SEED_BASE = 1000
DEFAULT_BOARD = {
    "start": [0.30, 0.20],
    "target": {"type": "circle", "center": [0.62, 0.66], "r": 0.05},
    "hazards": [{"type": "rect", "x0": 0.45, "y0": 0.35, "x1": 0.80, "y1": 0.45}],
    "markers": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
}
MAX_RULES = 2  # array capacity per config for conditionals and thresholds

_HEADER_TYPES = """\
#ifndef NM_CONFIG_TYPES_DEFINED
#define NM_CONFIG_TYPES_DEFINED
#include <stdint.h>

/* Channel index: A=0 B=1 C=2 D=3. Axis: u=0 v=1. */
enum { NM_ROLE_ACTUATOR = 1, NM_ROLE_INTERLOCK = 2, NM_ROLE_MODE = 3 };
enum { NM_EFF_GAIN_MULT = 1, NM_EFF_SIGN_FLIP = 2, NM_EFF_AXIS_SWAP = 3, NM_EFF_STOP = 4, NM_EFF_DROP = 5 };

typedef struct {
  uint8_t channel; uint8_t mode; uint8_t kind; double gain_mult;
} nm_conditional_t;

typedef struct {
  uint8_t axis; uint8_t op_gt; double value; uint8_t channel; uint8_t kind;
  double gain_mult; uint8_t drop_axis; double drop_amount;
} nm_threshold_t;

typedef struct {
  const char *id;
  uint8_t role[4];
  uint8_t act_axis[4]; int8_t act_sign[4]; double act_gain[4]; /* valid where role == ACTUATOR */
  uint8_t interlock_channel; uint8_t gated[4]; double interlock_threshold;
  uint8_t mode_channel; double mode_threshold; uint8_t mode_initial;
  uint8_t n_conditional; nm_conditional_t conditional[%(max)d];
  uint8_t n_thresholds; nm_threshold_t thresholds[%(max)d];
  uint8_t indicator_present; uint8_t indicator_shows_mode;
  double start_u; double start_v; /* servo units */
} nm_config_t;
#endif
""" % {"max": MAX_RULES}

_ROLE = {"actuator": "NM_ROLE_ACTUATOR", "interlock": "NM_ROLE_INTERLOCK", "mode": "NM_ROLE_MODE"}
_EFF = {
    "gain_mult": "NM_EFF_GAIN_MULT",
    "sign_flip": "NM_EFF_SIGN_FLIP",
    "axis_swap": "NM_EFF_AXIS_SWAP",
    "stop": "NM_EFF_STOP",
    "drop": "NM_EFF_DROP",
}
_AX = {"u": 0, "v": 1}


def _d(x: float) -> str:
    return repr(float(x))


def config_to_c(cfg: MachineConfig) -> str:
    idx = {ch: i for i, ch in enumerate(CHANNELS)}
    role = [_ROLE[cfg.roles[ch]] for ch in CHANNELS]
    axis = [str(_AX[cfg.actuators[ch].axis]) if ch in cfg.actuators else "0" for ch in CHANNELS]
    sign = [str(cfg.actuators[ch].sign) if ch in cfg.actuators else "0" for ch in CHANNELS]
    gain = [_d(cfg.actuators[ch].gain) if ch in cfg.actuators else "0.0" for ch in CHANNELS]
    gated = ["1" if ch in cfg.interlock.gates else "0" for ch in CHANNELS]
    conds = [
        f"{{{idx[c.channel]}, {c.mode}, {_EFF[c.effect.kind]}, {_d(c.effect.gain_mult or 1.0)}}}"
        for c in cfg.conditional
    ]
    ths = []
    for t in cfg.thresholds:
        drop = t.effect.drop
        ths.append(
            f"{{{_AX[t.axis]}, {1 if t.op == '>' else 0}, {_d(t.value)}, {idx[t.channel]}, {_EFF[t.effect.kind]}, "
            f"{_d(t.effect.gain_mult or 1.0)}, {_AX[drop.axis] if drop else 0}, {_d(drop.amount) if drop else '0.0'}}}"
        )
    if len(conds) > MAX_RULES or len(ths) > MAX_RULES:
        raise ValueError(f"{cfg.id}: more than {MAX_RULES} rules of one kind")
    pad_c = conds + ["{0, 0, 0, 1.0}"] * (MAX_RULES - len(conds))
    pad_t = ths + ["{0, 0, 0.0, 0, 0, 1.0, 0, 0.0}"] * (MAX_RULES - len(ths))
    return (
        "  {\n"
        f'    "{cfg.id}",\n'
        f"    {{{', '.join(role)}}},\n"
        f"    {{{', '.join(axis)}}}, {{{', '.join(sign)}}}, {{{', '.join(gain)}}},\n"
        f"    {idx[cfg.interlock.channel]}, {{{', '.join(gated)}}}, {_d(cfg.interlock.threshold)},\n"
        f"    {idx[cfg.mode.channel]}, {_d(cfg.mode.threshold)}, {cfg.mode.initial},\n"
        f"    {len(conds)}, {{{', '.join(pad_c)}}},\n"
        f"    {len(ths)}, {{{', '.join(pad_t)}}},\n"
        f"    {1 if cfg.indicator.present else 0}, {1 if cfg.indicator.shows == 'mode' else 0},\n"
        f"    {_d(cfg.start[0])}, {_d(cfg.start[1])}\n"
        "  }"
    )


def header(draws: list[Draw]) -> str:
    body = ",\n".join(config_to_c(d.config) for d in draws)
    seeds = ", ".join(str(d.seed) for d in draws)
    return (
        "/* Generated by scripts/export_configs.py (generator seeds: " + seeds + "). Do not edit.\n"
        " * Draw n (DIP switches) = NM_CONFIGS[n-1]; draw 0 = calibration mode.\n"
        " * Semantics: CONTRACTS.md section 3.2 and nomanual/core/machine.py (the reference). */\n"
        "#pragma once\n" + _HEADER_TYPES + f"\n#define NM_N_CONFIGS {len(draws)}\n"
        "static const nm_config_t NM_CONFIGS[NM_N_CONFIGS] = {\n" + body + "\n};\n"
    )


def golden_actions(cfg: MachineConfig, seed: int, n: int = 24) -> list[Action]:
    """A deterministic mixed sequence: single channels, interlock combos, mode presses,
    partial inputs and long holds, so every rule gets exercised."""
    rng = random.Random(seed)
    values = [-1.0, -0.6, -0.3, 0.0, 0.04, 0.3, 0.6, 1.0]
    out = []
    for i in range(n):
        if i % 5 == 0:
            ch = {cfg.mode.channel: 1.0}
        else:
            ch = {c: rng.choice(values) for c in CHANNELS}
            if i % 2:
                ch[cfg.interlock.channel] = 1.0
        out.append(Action(channels=ch, duration=rng.choice([0.2, 0.55, 1.0, 1.37, 2.0])))
    return out


def golden(draws: list[Draw]) -> dict:
    cases = []
    for d in draws:
        m = Machine(d.config)
        acts = golden_actions(d.config, d.seed)
        steps = []
        for i, a in enumerate(acts):
            if i == len(acts) // 2:
                m.swap()
            m.run(a.channels, a.duration)
            steps.append(
                {"action": a.model_dump(), "swap_before": i == len(acts) // 2,
                 "u": m.pos[0], "v": m.pos[1], "mode": m.mode, "led": m.led}
            )
        cases.append({"config": d.config.model_dump(mode="json"), "steps": steps})
    return {"tolerance": 1e-6, "cases": cases}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, help="board JSON (CONTRACTS §3.3); default: contract example")
    ap.add_argument("--n", type=int, default=8, help="number of draws (DIP 1..n)")
    ap.add_argument("--out", type=Path, default=Path("bench"))
    args = ap.parse_args(argv)
    board = Board.model_validate_json(args.board.read_text()) if args.board else Board.model_validate(DEFAULT_BOARD)
    draws = []
    for i in range(args.n):
        seed = RIG_SEED_BASE + i
        # Alternate plain and trap draws so the rig has both (traps need a rect hazard).
        d = generate(seed, trap=(i % 2 == 1), swap=True, board=board)
        draws.append(d)
        print(f"draw {i + 1}: {d.id} trap={d.trap} oracle legs={[len(r) for r in d.routes]}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "configs_export.h").write_text(header(draws), encoding="utf-8", newline="\n")
    (args.out / "configs_export.json").write_text(
        json.dumps({"board": board.model_dump(mode="json"),
                    "draws": [{"draw": i + 1, "seed": d.seed, "trap": d.trap, "config": d.config.model_dump(mode="json")}
                              for i, d in enumerate(draws)]}, indent=1),
        encoding="utf-8", newline="\n",
    )
    (args.out / "machine_golden.json").write_text(json.dumps(golden(draws)), encoding="utf-8", newline="\n")
    print(f"wrote {args.out / 'configs_export.h'}, configs_export.json, machine_golden.json")


if __name__ == "__main__":
    main()
