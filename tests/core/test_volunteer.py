"""The volunteer session (SPEC §8.3): chords, red-zone counting, the loop on a stub and on the fake rig."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from nomanual.core.board import Board
from nomanual.core.volunteer import (
    GIVE_UP,
    MAX_HOLD_S,
    RedZone,
    ScriptPad,
    SessionRecord,
    chord_action,
    run_session,
    save_session,
    summarize,
)

BOARD = Board.model_validate({
    "start": [0.3, 0.2], "target": {"type": "circle", "center": [0.62, 0.66], "r": 0.05},
    "hazards": [{"type": "rect", "x0": 0.45, "y0": 0.35, "x1": 0.8, "y1": 0.45}],
    "markers": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
})


def _record(**kw) -> SessionRecord:
    base = dict(participant="T1", condition="manual", machine="stub", manual=None, manual_sha256=None,
                board={}, started_at="", time_limit_s=30.0)
    base.update(kw)
    return SessionRecord(**base)


def test_chord_action_signs_and_conflicts():
    a = chord_action(["A+", "D-"])
    assert a is not None and a.channels["A"] == 1.0 and a.channels["D"] == -1.0
    assert a.channels["B"] == 0.0 and a.duration == MAX_HOLD_S
    b = chord_action(["A+", "A-", "C+"])
    assert b is not None and b.channels["A"] == 0.0 and b.channels["C"] == 1.0
    assert chord_action(["B+", "B-"]) is None
    assert chord_action([]) is None


def test_red_zone_counts_entries_not_frames():
    red = RedZone(BOARD)
    new, low = red.feed([(0.5, 0.30), (0.5, 0.38), (0.5, 0.40), (0.5, 0.30)])
    assert new == 1 and low < 0
    assert red.feed([(0.5, 0.30), (0.5, 0.31)])[0] == 0
    # a straight segment across the zone with both ends outside is an entry too
    new, _ = red.feed([(0.5, 0.30), (0.5, 0.50)])
    assert new == 1 and red.entries == 2
    assert red.feed([(0.2, 0.2)])[0] == 0


def test_script_pad_plays_presses_then_gives_up():
    t = [0.0]
    pad = ScriptPad([(0.0, 0.5, ["A+"]), (1.0, 1.2, ["B-", "C+"])], clock=lambda: t[0])
    assert pad.pressed() == frozenset({"A+"})
    t[0] = 1.1
    assert pad.pressed() == frozenset({"B-", "C+"})
    t[0] = 1.5
    assert pad.pressed() == frozenset()
    t[0] = 2.3
    assert GIVE_UP in pad.pressed()


class StubEnv:
    """Moves the tip right at `speed` per second while an action runs; `stop()` ends it."""

    def __init__(self, start=(0.3, 0.2), speed=0.2):
        self.pos = list(start)
        self.speed = speed
        self.stops = 0
        self.steps: list = []
        self._stop = threading.Event()
        self.tracker = SimpleNamespace(update=lambda img: None)
        self.stop_reason = None

    def _obs(self):
        return SimpleNamespace(pos=tuple(self.pos), confidence=1.0)

    def reset(self):
        return self._obs()

    def stop(self):
        self.stops += 1
        self._stop.set()

    def step(self, action):
        self._stop.clear()
        before = self._obs()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < action.duration and not self._stop.wait(0.005):
            pass
        dt = time.perf_counter() - t0
        self.pos[0] += self.speed * dt * action.channels["A"]
        self.pos[1] += self.speed * dt * action.channels["B"]
        self.steps.append((dict(action.channels), dt))
        return SimpleNamespace(before=before, after=self._obs(), status="stopped" if dt < action.duration else "done")


def test_release_stops_the_action_and_success_ends_the_session(tmp_path):
    board = Board.model_validate({**BOARD.model_dump(mode="json"),
                                  "target": {"type": "circle", "center": [0.4, 0.24], "r": 0.05}})
    env = StubEnv()
    # first a short tap of B (moves up), then A+ held long enough to reach x = 0.4 (0.5 s at 0.2/s)
    pad = ScriptPad([(0.1, 0.35, ["B+"]), (0.6, 1.1, ["A+"]), (1.6, 3.0, ["A+"])])
    rec = run_session(env, board, pad, _record(), log=lambda m: None)
    assert rec.outcome == "success" and rec.success
    assert len(rec.actions) == 2, rec.actions
    first, second = rec.actions
    assert first.inputs == ["B+"] and 0.15 <= first.held_s <= 0.4
    assert env.steps[0][1] < 0.5, "the release must cut the action short"
    assert second.inputs == ["A+"] and 0.35 <= second.held_s <= 0.65
    assert rec.red_entries == 0 and rec.time_s is not None and rec.time_s < 2.0
    path = save_session(rec, tmp_path / "run", tmp_path / "study.jsonl")
    assert path.exists() and (tmp_path / "study.jsonl").read_text().count("\n") == 1


def test_nothing_pressed_then_give_up_is_not_started():
    env = StubEnv()
    rec = run_session(env, BOARD, ScriptPad([(5.0, 5.0, [])]), _record(time_limit_s=1.0), log=lambda m: None)
    assert rec.outcome == "not started" and rec.time_s is None and not rec.actions


def test_summary_table_counts_per_condition():
    rows = [
        {"participant": "P1", "condition": "manual", "outcome": "success", "success": True, "time_s": 40.0,
         "red_entries": 0, "actions": 12, "machine": "draw1"},
        {"participant": "P2", "condition": "none", "outcome": "time limit", "success": False, "time_s": 300.0,
         "red_entries": 3, "actions": 60, "machine": "draw1"},
    ]
    text = summarize(rows)
    assert "| manual | 1 | 1/1 | 40 s | 0 | 0/1 | 12 |" in text
    assert "| none | 1 | 0/1 | – | 3 | 1/1 | 60 |" in text


def test_session_on_the_fake_rig_reaches_a_target_right_of_the_start(tmp_path):
    """Draw 1: A moves the tip right while C is held. A volunteer taps A+ and C+ together."""
    from nomanual.rig.host_engine import HostBuildError, build_host_binary

    try:
        build_host_binary()
    except HostBuildError as exc:
        pytest.skip(f"firmware host build unavailable: {exc}")
    from nomanual.rig.fake_rig import build_fake_rig

    board = Board.model_validate({**BOARD.model_dump(mode="json"),
                                  "target": {"type": "circle", "center": [0.42, 0.2], "r": 0.04}})
    presses = [(0.3 + 1.6 * k, 0.3 + 1.6 * k + 0.35, ["A+", "C+"]) for k in range(8)]
    rig = build_fake_rig(None, draw=1, time_scale=1.0, run_dir=tmp_path / "run")
    try:
        rec = run_session(rig.env, board, ScriptPad(presses), _record(time_limit_s=40.0), log=lambda m: None)
    finally:
        rig.close()
    assert rec.outcome == "success", [(a.status, a.after) for a in rec.actions]
    assert rec.red_entries == 0
    assert all(a.status in ("stopped", "done") for a in rec.actions)
    assert all(a.after[0] > a.before[0] for a in rec.actions if a.before and a.after)
    assert all(a.fixes > 0 for a in rec.actions)
