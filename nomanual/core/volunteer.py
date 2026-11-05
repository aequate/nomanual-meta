"""The human manual study (SPEC §8.3): a volunteer drives the rig with a hand controller.

A session is one volunteer, one condition (`manual`: they hold the generated manual; `none`:
they do not), one task: bring the tip into the green target without touching the red zone.

- **Input.** The pad reports which logical inputs are held: `A+ A- … D+ D-` and `give_up`.
  Pressing controls starts an action with those channels at ±1 (a control held both ways is
  left at 0). Inputs pressed within `CHORD_S` of the first one join the chord, so "hold A+ and
  D-" is one action. The action runs while every input of the chord is held, up to `MAX_HOLD_S`;
  releasing any of them sends `X`. Every control must be released before the next action, so a
  long hold never repeats by itself (a mode press stays one press).
- **No shield.** The red zone is the thing being measured, so nothing stops a volunteer from
  entering it. The tip is above the board, the zone is tape; the physical limits are the
  firmware's (soft limits, rate cap, travel box, watchdog), `RigEnv`'s off-board stop and the
  servo cut-off.
- **Measured** from the tracked tip: every camera fix during each action and the still before
  and after. A red-zone entry is a change from outside to inside the hazard along that sequence
  (fixes and the straight segments between them). Success is a still inside the target. The
  timer starts at the first press and stops at the end of the action that reached the target.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nomanual.core.board import Board
from nomanual.core.env import CHANNELS, Action

CONTROL_INPUTS = tuple(f"{c}{s}" for c in CHANNELS for s in "+-")
GIVE_UP = "give_up"
INPUTS = (*CONTROL_INPUTS, GIVE_UP)
CONDITIONS = ("manual", "none")

CHORD_S = 0.15
"""Inputs pressed this soon after the first one belong to the same action."""
MAX_HOLD_S = 2.0
"""The longest action the firmware runs (CONTRACTS §4: 0.2..2.0 s)."""
MIN_HOLD_S = 0.2
"""A tap still moves for this long: the firmware's shortest action."""
POLL_S = 0.01
STOP_RESEND_S = 0.05
"""After a release, `X` is sent again at this interval until the action has ended (the release
may come before the `R` line went out)."""


class Pad(Protocol):
    def pressed(self) -> frozenset[str]:
        """The logical inputs held right now (a subset of `INPUTS`)."""
        ...


class ScriptPad:
    """A pad that plays back `(start_s, end_s, inputs)` presses from its first poll (tests, rehearsal)."""

    def __init__(self, presses: Iterable[tuple[float, float, Iterable[str]]],
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.presses = [(a, b, frozenset(i)) for a, b, i in presses]
        self.clock = clock
        self.t0: float | None = None

    def pressed(self) -> frozenset[str]:
        now = self.clock()
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        out: set[str] = set()
        for a, b, inputs in self.presses:
            if a <= t < b:
                out |= inputs
        if self.presses and t >= max(b for _, b, _ in self.presses) + 1.0:
            out.add(GIVE_UP)  # the script is over
        return frozenset(out)


def chord_action(inputs: Iterable[str]) -> Action | None:
    """The action for a chord of control inputs; None when no control ends up non-zero."""
    held = set(inputs)
    channels = {}
    for c in CHANNELS:
        v = (1.0 if f"{c}+" in held else 0.0) - (1.0 if f"{c}-" in held else 0.0)
        if v:
            channels[c] = v
    if not channels:
        return None
    return Action(channels=channels, duration=MAX_HOLD_S)


class RecordingTracker:
    """Wraps `RigEnv.tracker`: keeps every good fix `(t, x, y)` it returns, for `take()`."""

    def __init__(self, inner: Any, clock: Callable[[], float] = time.perf_counter) -> None:
        self._inner = inner
        self._clock = clock
        self._lock = threading.Lock()
        self._fixes: list[tuple[float, float, float]] = []

    def update(self, image: Any) -> Any:
        fix = self._inner.update(image)
        if getattr(fix, "ok", False) and fix.pos is not None:
            with self._lock:
                self._fixes.append((self._clock(), float(fix.pos[0]), float(fix.pos[1])))
        return fix

    def take(self) -> list[tuple[float, float, float]]:
        with self._lock:
            out, self._fixes = self._fixes, []
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class RedZone:
    """Red-zone bookkeeping over the whole session."""

    board: Board
    inside: bool = False
    entries: int = 0
    min_clearance: float = float("inf")

    def feed(self, points: list[tuple[float, float]]) -> tuple[int, float]:
        """Add a stretch of tip positions (in order). Returns (new entries, min clearance here)."""
        new, low = 0, float("inf")
        prev: tuple[float, float] | None = None
        for p in points:
            d = float(self.board.hazard_distance(p)[0]) if self.board.hazards else float("inf")
            seg = d if prev is None or not self.board.hazards else self.board.segment_hazard_distance(prev, p)
            low = min(low, d, seg)
            now_inside = d < 0
            if not self.inside and (now_inside or seg < 0):
                new += 1
            self.inside = now_inside
            prev = p
        self.entries += new
        self.min_clearance = min(self.min_clearance, low)
        return new, low


@dataclass
class ActionRecord:
    i: int
    t: float
    """Seconds since the first press, at this press."""
    inputs: list[str]
    channels: dict[str, float]
    held_s: float
    """How long the volunteer held the chord (capped by MAX_HOLD_S)."""
    status: str
    before: list[float] | None
    after: list[float] | None
    fixes: int
    red_entries: int
    min_clearance: float | None
    laptop_stop: str | None = None


@dataclass
class SessionRecord:
    participant: str
    condition: str
    machine: str
    manual: str | None
    manual_sha256: str | None
    board: dict
    started_at: str
    time_limit_s: float
    outcome: str = "not started"
    """'success', 'gave up', 'time limit', 'interrupted' or 'error: …'."""
    success: bool = False
    time_s: float | None = None
    """From the first press to the end of the last action (the one that reached the target on success)."""
    red_entries: int = 0
    min_clearance: float | None = None
    actions: list[ActionRecord] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pos(obs: Any) -> tuple[float, float] | None:
    if obs is None or getattr(obs, "confidence", 0.0) <= 0.0 or obs.pos is None:
        return None
    return float(obs.pos[0]), float(obs.pos[1])


class _RunStarted:
    """When the current action reached the rig client. An env with no `client` (a stub) counts
    as started when its step begins."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.t = 0.0
        self.hooked = False
        self._ev = threading.Event()

    def mark(self) -> None:
        self.hooked = True
        self.t = self.clock()
        self._ev.set()

    def clear(self) -> None:
        self._ev.clear()
        if not self.hooked:
            self.t = self.clock()
            self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set()


def run_session(
    env: Any,
    board: Board,
    pad: Pad,
    record: SessionRecord,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    on_update: Callable[[dict], None] | None = None,
    log: Callable[[str], None] = print,
) -> SessionRecord:
    """Run one volunteer session on `env` (a `RigEnv`, already built). Resets the rig first."""
    tracker = env.tracker if isinstance(env.tracker, RecordingTracker) else RecordingTracker(env.tracker, clock)
    env.tracker = tracker
    red = RedZone(board)
    state: dict[str, Any] = {"t": None, "status": "ready: press any control to start",
                             "red": 0, "condition": record.condition, "participant": record.participant}

    def update(**kw: Any) -> None:
        state.update(kw)
        if on_update is not None:
            on_update(dict(state))

    run_started = _RunStarted(clock)
    client = getattr(env, "client", None)
    inner_run = getattr(client, "run", None)
    own_run = inner_run is not None and "run" in vars(client)
    if inner_run is not None:
        def timed_run(*a: Any, **kw: Any) -> Any:
            run_started.mark()
            return inner_run(*a, **kw)
        client.run = timed_run

    obs = env.reset()
    tracker.take()
    pos = _pos(obs)
    if pos is not None:
        red.feed([pos])
    t_first: float | None = None
    t_last_end: float | None = None
    record.outcome = "not started"
    update()

    def elapsed() -> float | None:
        return None if t_first is None else clock() - t_first

    try:
        while True:
            update(t=elapsed())
            if t_first is not None and clock() - t_first >= record.time_limit_s:
                record.outcome = "time limit"
                break
            held = pad.pressed()
            if GIVE_UP in held:
                record.outcome = "gave up" if t_first is not None else "not started"
                break
            ctrl = held & set(CONTROL_INPUTS)
            if not ctrl:
                sleep(POLL_S)
                continue

            t_press = clock()
            chord = set(ctrl)
            while clock() - t_press < CHORD_S:
                now_held = pad.pressed() & set(CONTROL_INPUTS)
                if not now_held:
                    break
                chord |= now_held
                sleep(POLL_S)
            action = chord_action(chord)
            if action is None:
                update(status="that combination does nothing: release and try again")
                while pad.pressed() & set(CONTROL_INPUTS):
                    sleep(POLL_S)
                continue
            if t_first is None:
                t_first = t_press

            label = " ".join(sorted(chord))
            update(t=elapsed(), status=f"running {label}")
            out: dict[str, Any] = {}

            def work(action: Action = action, out: dict[str, Any] = out) -> None:
                try:
                    out["res"] = env.step(action)
                except Exception as exc:  # noqa: BLE001 - reported in the record
                    out["exc"] = exc

            th = threading.Thread(target=work, name="volunteer-step", daemon=True)
            run_started.clear()
            th.start()
            released_at: float | None = None
            last_x = 0.0
            while th.is_alive():
                if released_at is None and not chord <= pad.pressed():
                    released_at = clock()
                # The rig looks before it moves, so the action starts some time after the press.
                # The volunteer's hold is replayed from that start; stopping at the release
                # would cancel a short press before it ran.
                if (released_at is not None and run_started.is_set()
                        and clock() - run_started.t >= max(MIN_HOLD_S, released_at - t_press)
                        and clock() - last_x >= STOP_RESEND_S):
                    env.stop()
                    last_x = clock()
                update(t=elapsed())
                sleep(POLL_S)
            th.join()
            t_last_end = clock()
            held_s = min(MAX_HOLD_S, (released_at or t_last_end) - t_press)
            if "exc" in out:
                record.outcome = f"error: {type(out['exc']).__name__}: {out['exc']}"
                log(record.outcome)
                break
            res = out["res"]
            before, after = _pos(res.before), _pos(res.after)
            fixes = tracker.take()
            pts = ([before] if before else []) + [(x, y) for _, x, y in fixes] + ([after] if after else [])
            new, low = red.feed(pts) if pts else (0, float("inf"))
            rec = ActionRecord(
                i=len(record.actions), t=round(t_press - t_first, 3), inputs=sorted(chord),
                channels={k: v for k, v in action.channels.items() if v}, held_s=round(held_s, 3),
                status=res.status, before=list(before) if before else None,
                after=list(after) if after else None, fixes=len(fixes), red_entries=new,
                min_clearance=None if low == float("inf") else round(low, 4),
                laptop_stop=getattr(env, "stop_reason", None),
            )
            record.actions.append(rec)
            where = f"({after[0]:.2f}, {after[1]:.2f})" if after else "(tip not seen)"
            log(f"{rec.t:7.1f} s  {label:<12} held {held_s:.2f} s  {res.status:<8} -> {where}"
                + (f"  RED ZONE x{new}" if new else ""))
            update(t=elapsed(), red=red.entries, status=f"{label}: {res.status}")
            if after is not None and board.target.contains(after):
                record.outcome = "success"
                record.success = True
                break
            while pad.pressed() & set(CONTROL_INPUTS):
                update(t=elapsed())
                sleep(POLL_S)
            update(status="ready")
    except KeyboardInterrupt:
        record.outcome = "interrupted"
    finally:
        try:
            env.stop()
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
        if own_run:
            client.run = inner_run
        elif inner_run is not None:
            del client.run
    if t_first is not None:
        end = t_last_end if record.success and t_last_end is not None else clock()
        record.time_s = round(end - t_first, 2)
    record.red_entries = red.entries
    record.min_clearance = None if red.min_clearance == float("inf") else round(red.min_clearance, 4)
    update(t=record.time_s, red=red.entries, status=record.outcome, done=True)
    return record


def save_session(record: SessionRecord, run_dir: Path, study: Path | None) -> Path:
    """Write `<run_dir>/session.json`; append a one-line summary to `study` (JSONL)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "session.json"
    path.write_text(json.dumps(record.to_json(), indent=1))
    if study is not None:
        study.parent.mkdir(parents=True, exist_ok=True)
        line = {k: v for k, v in record.to_json().items() if k not in ("actions", "board")}
        line["actions"] = len(record.actions)
        line["session"] = str(path)
        with study.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    return path


def summarize(rows: list[dict]) -> str:
    """Markdown summary per condition. Small samples: counts and medians only, no tests."""
    lines = [("| condition | volunteers | reached target | median time (successes) | red-zone entries (total) "
              "| volunteers with ≥1 entry | median actions |"),
             "|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        rs = [r for r in rows if r.get("condition") == cond]
        if not rs:
            lines.append(f"| {cond} | 0 | – | – | – | – | – |")
            continue
        ok = [r for r in rs if r.get("success")]
        times = [r["time_s"] for r in ok if r.get("time_s") is not None]
        med_t = f"{statistics.median(times):.0f} s" if times else "–"
        acts = [r.get("actions", 0) for r in rs]
        lines.append(
            f"| {cond} | {len(rs)} | {len(ok)}/{len(rs)} | {med_t} | {sum(r.get('red_entries', 0) for r in rs)} "
            f"| {sum(1 for r in rs if r.get('red_entries', 0) > 0)}/{len(rs)} | {statistics.median(acts):.0f} |")
    lines += ["", "| participant | condition | outcome | time | red-zone entries | actions | machine |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = f"{r['time_s']:.0f} s" if r.get("time_s") is not None else "–"
        lines.append(f"| {r['participant']} | {r['condition']} | {r['outcome']} | {t} | {r.get('red_entries', 0)} "
                     f"| {r.get('actions', 0)} | {r.get('machine', '')} |")
    return "\n".join(lines) + "\n"
