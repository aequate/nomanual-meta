"""Benchmark sets (SPEC §8.1): a dev set for tuning and a held-out set for the final report.

Each set is a list of machines drawn by `generator.generate` from fixed seeds, stored in full
(configuration and board) as canonical JSON under `bench/`, so later generator changes cannot
change a frozen set. The sha256 of each file is recorded in `bench/README.md`, and `load`
refuses a file whose hash differs from the recorded one.

Composition: in seed order, every second machine has a threshold trap and every third a
rewire after the first task (dev 20: 10 traps, 7 rewires; held-out 30: 15 traps, 10 rewires).

The held-out set is loaded only with `allow_heldout=True`: the human decides when it runs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nomanual.core.board import Board
from nomanual.core.generator import generate
from nomanual.core.machine import MachineConfig

SetName = Literal["dev", "heldout"]
SEEDS: dict[str, range] = {"dev": range(1000, 1020), "heldout": range(5000, 5030)}
BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"


class HeldOutLocked(PermissionError):
    """The held-out set was requested without `allow_heldout=True`."""


class HashMismatch(ValueError):
    """A set file does not match the sha256 recorded in bench/README.md."""


@dataclass(frozen=True)
class Item:
    set: str
    index: int
    seed: int
    trap: bool
    swap: bool
    config: MachineConfig
    board: Board

    @property
    def name(self) -> str:
        return f"{self.set}-{self.index:02d}"


def composition(name: SetName) -> list[tuple[int, bool, bool]]:
    """(seed, trap, swap) per machine."""
    return [(seed, i % 2 == 0, i % 3 == 0) for i, seed in enumerate(SEEDS[name])]


def draw(name: SetName) -> list[Item]:
    return [Item(name, i, seed, trap, swap, (d := generate(seed, trap=trap, swap=swap)).config, d.board)
            for i, (seed, trap, swap) in enumerate(composition(name))]


def fresh(seeds: Sequence[int]) -> list[Item]:
    """Machines from generator seeds outside both sets (set "fresh"), with the sets' composition
    rule: a threshold trap on every second machine and a rewire on every third. Refuses a dev or
    held-out seed (checked against the seed ranges; no set file is opened)."""
    taken = [s for s in seeds if any(s in SEEDS[name] for name in SEEDS)]
    if taken:
        raise ValueError(f"seeds {taken} belong to the dev or held-out set; fresh machines need other seeds")
    return [Item("fresh", i, seed, i % 2 == 0, i % 3 == 0, (d := generate(seed, trap=i % 2 == 0, swap=i % 3 == 0)).config,
                 d.board) for i, seed in enumerate(seeds)]


def dumps(items: list[Item]) -> bytes:
    rows = [{"index": it.index, "seed": it.seed, "trap": it.trap, "swap": it.swap,
             "config": it.config.model_dump(mode="json"), "board": it.board.model_dump(mode="json")} for it in items]
    return (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(data: bytes) -> str:
    """Of the file with LF line ends (a Windows checkout may turn the final newline into CRLF)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def recorded_hashes(readme: Path) -> dict[str, str]:
    """`| <set> | <machines> | <traps> | <rewires> | <sha256> |` rows of bench/README.md."""
    if not readme.exists():
        return {}
    return {m.group(1): m.group(2) for m in
            re.finditer(r"^\|\s*`?(dev|heldout)`?\s*\|.*?\|\s*`?([0-9a-f]{64})`?\s*\|\s*$", readme.read_text(), re.M)}


def load(name: SetName, bench_dir: Path = BENCH_DIR, allow_heldout: bool = False) -> list[Item]:
    if name == "heldout" and not allow_heldout:
        raise HeldOutLocked("the held-out set runs only when the human decides (pass allow_heldout=True)")
    path = bench_dir / f"{name}.json"
    data = path.read_bytes()
    want = recorded_hashes(bench_dir / "README.md").get(name)
    if want is None or sha256(data) != want:
        raise HashMismatch(f"{path}: sha256 {sha256(data)} does not match bench/README.md ({want})")
    return [Item(name, r["index"], r["seed"], r["trap"], r["swap"], MachineConfig.model_validate(r["config"]),
                 Board.model_validate(r["board"])) for r in json.loads(data)]
