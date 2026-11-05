"""Run event log (CONTRACTS §6): Pydantic models for every event kind, a JSONL writer, a reader.

One run = `runs/<run_id>/events.jsonl` + `runs/<run_id>/frames/*.jpg` + `runs/<run_id>/manual.md`.
Every line is `{"t", "run", "seq", "kind", "data"}`; `data` is validated against the model
registered for `kind` in `EVENT_MODELS`. The dashboard reads these files, so the schema here
must match the contract exactly; optional additive fields are noted in the CONTRACTS change log.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nomanual.core.board import Board
from nomanual.core.env import Action, Observation, StepResult

RuleState = Literal["supported", "plausible", "contradicted", "ambiguous"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStart(_Strict):
    env: Literal["sim", "rig"]
    board: Board
    config_id_hidden: str
    model: str
    provider: str


class ProposedHypothesis(_Strict):
    id: str
    text: str
    program: dict[str, Any]


class Proposal(_Strict):
    source: Literal["muse", "enum", "repair"]
    hypotheses: list[ProposedHypothesis]
    reasoning: str = Field(max_length=300)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_s: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)

    @field_validator("reasoning", mode="before")
    @classmethod
    def _trim(cls, v: object) -> object:
        # The contract caps reasoning at 300 chars; trim instead of failing a whole run.
        if isinstance(v, str) and len(v) > 300:
            return v[:299] + "…"
        return v


class ObservableProposed(_Strict):
    name: str
    description: str
    evidence_frames: list[str]
    source: Literal["muse"]


class ObservableTracked(_Strict):
    name: str
    ok: bool


class HypothesisItem(_Strict):
    id: str
    text: str
    program: dict[str, Any]
    state: RuleState
    score: float
    killed_by: int | None = None


class RuleItem(_Strict):
    """Additive (change log 2026-10-27): one rule (component of a program) and its state."""

    id: str
    text: str
    state: RuleState
    evidence: list[int] = Field(default_factory=list)
    """Experiment seqs that put this rule at risk and that it predicted correctly."""


class Hypotheses(_Strict):
    items: list[HypothesisItem]
    total: int | None = None
    """Additive: total hypotheses in the pool (items may be a truncated, ranked subset)."""
    rules: list[RuleItem] | None = None
    """Additive: rule-level states across surviving hypotheses (SPEC §3.3)."""


class PlanCandidate(_Strict):
    action: Action
    value: float
    why: str


class Plan(_Strict):
    candidates: list[PlanCandidate]
    chosen: int
    why: str


class ShieldEvent(_Strict):
    action: Action
    verdict: Literal["accept", "shrink", "reject"]
    reason: str
    final: Action | None = None


class ActionEvent(_Strict):
    seq: int
    action: Action


class ResultEvent(StepResult):
    residuals: dict[str, float] = Field(default_factory=dict)


class Alarm(_Strict):
    stat: float
    threshold: float
    suspect_rules: list[str]
    cause: Literal["drift", "collapse"] = "drift"
    """drift: the change monitor crossed its threshold on the leader's residuals; collapse: an
    experiment refuted the leader and every survivor, no single edit of the leader explained the
    data, and the monitor, still watching the refuted leader, then crossed its threshold."""


class Repair(_Strict):
    edits: list[str]
    accepted: bool


Cell = tuple[float, float, float, float]


class Knowledge(_Strict):
    verified_cells: list[Cell]
    unknown_cells: list[Cell]
    unsafe_cells: list[Cell]
    thresholds: list[dict[str, Any]]
    path: list[tuple[float, float]]


class TaskEvent(_Strict):
    name: str
    status: Literal["start", "success", "fail"]
    experiments: int = Field(ge=0)


class ManualEvent(_Strict):
    path: str
    claims: int = Field(ge=0)
    cited: int = Field(ge=0)


class RunEnd(_Strict):
    status: str
    experiments: int = Field(ge=0)
    tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    wall_s: float = Field(ge=0.0)


EVENT_MODELS: dict[str, type[BaseModel]] = {
    "run_start": RunStart,
    "observation": Observation,
    "proposal": Proposal,
    "observable_proposed": ObservableProposed,
    "observable_tracked": ObservableTracked,
    "hypotheses": Hypotheses,
    "plan": Plan,
    "shield": ShieldEvent,
    "action": ActionEvent,
    "result": ResultEvent,
    "alarm": Alarm,
    "repair": Repair,
    "knowledge": Knowledge,
    "task": TaskEvent,
    "manual": ManualEvent,
    "run_end": RunEnd,
}

EventKind = Literal[
    "run_start",
    "observation",
    "proposal",
    "observable_proposed",
    "observable_tracked",
    "hypotheses",
    "plan",
    "shield",
    "action",
    "result",
    "alarm",
    "repair",
    "knowledge",
    "task",
    "manual",
    "run_end",
]


class Event(_Strict):
    t: float
    run: str
    seq: int = Field(ge=0)
    kind: EventKind
    data: BaseModel

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> Event:
        kind = raw.get("kind")
        model = EVENT_MODELS.get(kind) if isinstance(kind, str) else None
        if model is None:
            raise ValueError(f"unknown event kind {kind!r}")
        data = model.model_validate(raw.get("data", {}))
        return cls(t=raw["t"], run=raw["run"], seq=raw["seq"], kind=kind, data=data)

    def to_json(self) -> str:
        return json.dumps(
            {
                "t": round(self.t, 4),
                "run": self.run,
                "seq": self.seq,
                "kind": self.kind,
                "data": self.data.model_dump(mode="json"),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )


def new_run_id(prefix: str = "r", now: datetime | None = None, runs_root: Path | None = None) -> str:
    """`r-YYYYMMDD-NNN`, numbered after any existing runs of the same day under `runs_root`."""
    day = (now or datetime.now()).strftime("%Y%m%d")
    n = 1
    if runs_root is not None and runs_root.exists():
        taken = {p.name for p in runs_root.iterdir()}
        while f"{prefix}-{day}-{n:03d}" in taken:
            n += 1
    return f"{prefix}-{day}-{n:03d}"


class EventWriter:
    """Append-only JSONL writer for one run. Each line is flushed so a live tail sees it."""

    def __init__(
        self,
        run_dir: Path | str,
        run_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        self.path = self.run_dir / "events.jsonl"
        self._t0 = time.monotonic()
        self._clock = clock or (lambda: time.monotonic() - self._t0)
        self._seq = 0
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")

    @property
    def frames_dir(self) -> Path:
        d = self.run_dir / "frames"
        d.mkdir(exist_ok=True)
        return d

    @property
    def seq(self) -> int:
        """Sequence number the next event will get."""
        return self._seq

    def emit(self, kind: str, data: BaseModel | dict[str, Any]) -> Event:
        model = EVENT_MODELS.get(kind)
        if model is None:
            raise ValueError(f"unknown event kind {kind!r}")
        if isinstance(data, BaseModel) and not isinstance(data, model):
            data = data.model_dump(mode="json")
        payload = data if isinstance(data, model) else model.model_validate(data)
        event = Event(t=float(self._clock()), run=self.run_id, seq=self._seq, kind=kind, data=payload)
        self._fh.write(event.to_json() + "\n")
        self._fh.flush()
        self._seq += 1
        return event

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_events(path: Path | str) -> Iterator[Event]:
    """Yield validated events from a JSONL file (or a run directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / "events.jsonl"
    with p.open(encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            if lineno == len(lines):
                return  # a run cut off mid-write: keep the complete events before it (audit F9)
            raise ValueError(f"{p}:{lineno}: invalid event: {exc}") from exc
        try:
            yield Event.parse(raw)
        except Exception as exc:  # noqa: BLE001 - re-raised with location
            raise ValueError(f"{p}:{lineno}: invalid event: {exc}") from exc


def read_events(path: Path | str) -> list[Event]:
    return list(iter_events(path))
