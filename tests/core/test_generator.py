import json
import math
from pathlib import Path

import pytest

from nomanual.core.board import THRESHOLD_GRID, Board, servo_to_board
from nomanual.core.generator import LAYOUT_CLEAR, Draw, check_solvable, generate
from nomanual.core.machine import Machine, MachineConfig

FAST_SEEDS = [(0, True, True), (1, False, False), (3, False, True)]


@pytest.fixture(scope="module")
def draws() -> list[Draw]:
    return [generate(s, trap=t, swap=w) for s, t, w in FAST_SEEDS]


def test_same_seed_same_draw(draws: list[Draw]) -> None:
    again = generate(0, trap=True, swap=True)
    assert again.config == draws[0].config and again.board == draws[0].board


def test_structure(draws: list[Draw]) -> None:
    for d in draws:
        cfg = d.config
        axes = {a.axis for a in cfg.actuators.values()}
        assert axes == {"u", "v"}  # one actuator per servo
        assert cfg.mode.initial == 0
        assert all(c.mode == 1 for c in cfg.conditional)
        # The mode always has an observable consequence (a conditional or the light).
        assert cfg.conditional or cfg.indicator.present
        for t in cfg.thresholds:
            assert round(float(servo_to_board(t.value)), 6) in THRESHOLD_GRID
        assert d.board.hazard_distance(d.board.start)[0] >= LAYOUT_CLEAR
        assert d.board.hazard_distance(d.board.target.center)[0] >= LAYOUT_CLEAR
        assert d.board.segment_hazard_distance(d.board.start, d.board.target.center) <= 0.0


def test_board_start_matches_machine_start(draws: list[Draw]) -> None:
    for d in draws:
        m = Machine(d.config)
        assert (float(servo_to_board(m.pos[0])), float(servo_to_board(m.pos[1]))) == pytest.approx(d.board.start, abs=1e-5)


def test_trap_threshold_pushes_toward_hazard(draws: list[Draw]) -> None:
    d = draws[0]
    assert d.trap and len(d.config.thresholds) == 1
    t = d.config.thresholds[0]
    assert t.effect.kind == "drop"
    assert d.config.actuators[t.channel].axis == t.axis  # the progress actuator triggers it


def test_oracle_routes_replay_safely(draws: list[Draw]) -> None:
    for d in draws:
        m = Machine(d.config)
        goals = [(d.board.target.center, d.board.target.r), (d.board.start, 0.03), (d.board.target.center, d.board.target.r)]
        for leg, (route, (goal, r)) in enumerate(zip(d.routes, goals)):
            if d.swap and leg == 1:
                m.swap()
            for a in route:
                tr = m.run(a.channels, a.duration, record=True)
                pts = [(float(servo_to_board(u)), float(servo_to_board(v))) for u, v in [tr.start, *tr.path]]
                assert d.board.hazard_distance(pts).min() >= 0.03
            pos = (float(servo_to_board(m.pos[0])), float(servo_to_board(m.pos[1])))
            assert math.dist(pos, goal) <= r


def test_unsolvable_detected() -> None:
    d = generate(1)
    # Put the target inside the hazard: no route can end there safely.
    hz = d.board.hazards[0]
    bad = d.board.model_copy(update={"target": d.board.target.model_copy(update={"center": ((hz.x0 + hz.x1) / 2, (hz.y0 + hz.y1) / 2)})})
    from nomanual.core.generator import Unsolvable

    with pytest.raises(Unsolvable):
        check_solvable(d.config, bad, swap=False)


def test_fixed_board_draws_start_at_board_start() -> None:
    board = Board.model_validate({
        "start": [0.30, 0.20], "target": {"type": "circle", "center": [0.62, 0.66], "r": 0.05},
        "hazards": [{"type": "rect", "x0": 0.45, "y0": 0.35, "x1": 0.80, "y1": 0.45}],
    })
    d = generate(1001, trap=True, board=board)
    assert d.board == board
    assert (float(servo_to_board(d.config.start[0])), float(servo_to_board(d.config.start[1]))) == pytest.approx((0.30, 0.20), abs=1e-5)


def test_exported_files_are_consistent() -> None:
    root = Path(__file__).resolve().parents[2] / "bench"
    exported = json.loads((root / "configs_export.json").read_text())
    header = (root / "configs_export.h").read_text()
    golden = json.loads((root / "machine_golden.json").read_text())
    assert f"#define NM_N_CONFIGS {len(exported['draws'])}" in header
    for item, case in zip(exported["draws"], golden["cases"]):
        cfg = MachineConfig.model_validate(item["config"])
        assert f'"{cfg.id}"' in header
        # Golden traces replay exactly on the reference machine.
        m = Machine(cfg)
        for step in case["steps"]:
            if step["swap_before"]:
                m.swap()
            a = step["action"]
            m.run(a["channels"], a["duration"])
            assert (m.pos[0], m.pos[1], m.mode, m.led) == (step["u"], step["v"], step["mode"], step["led"])


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(100, 110))
def test_many_seeds_generate(seed: int) -> None:
    d = generate(seed, trap=seed % 2 == 0, swap=seed % 3 == 0)
    assert len(d.routes) == 3
