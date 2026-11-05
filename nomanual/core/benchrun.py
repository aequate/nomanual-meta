"""Benchmark runner: every (arm, machine) pair once, resumable, in at most two worker processes.

Each run is a fresh worker process (`maxtasksperchild=1`), so memory goes back to the system
after every run. Records go to a JSONL file as they finish; a rerun skips the pairs already
there. A rewire item gets its rewire between the first and the second task (`inject_swap`).
Frames are rendered only for the model arms (the vision path needs them) and are deleted after
the run unless `keep_runs`. SIGTERM to the runner stops its workers too: an orphaned worker
would finish its run and delete that run's folder, under a new runner using the same folders.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import signal
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nomanual.core import bench
from nomanual.core.agent import Agent
from nomanual.core.baselines import MODEL_ARMS, Arm, LinearAgent, agent_config
from nomanual.core.events import EventWriter
from nomanual.core.models import CallLog, FakeModel, ReasoningModel, make_model
from nomanual.core.proposer import FAKE_LABEL, ModelProposer, fake_responder
from nomanual.core.scoring import score_run
from nomanual.core.sim import SimEnv

TASKS = ("reach", "park", "reach")
MAX_WORKERS = 2


@dataclass(frozen=True)
class Job:
    arm: Arm
    item: bench.Item
    budget: int
    provider: str
    runs_root: Path
    keep_runs: bool


def model_for(provider: str, seed: int) -> tuple[ReasoningModel, str]:
    """(model, label). "fake" is the heuristic stand-in; anything else is a live provider."""
    if provider == "fake":
        return FakeModel(fake_responder(seed=seed)), FAKE_LABEL
    m = make_model(provider)
    return m, f"{provider} {m.model}"


def run_job(job: Job) -> dict[str, Any]:
    item = job.item
    run_dir = job.runs_root / f"{job.arm}_{item.name}"
    shutil.rmtree(run_dir, ignore_errors=True)
    model_arm = job.arm in MODEL_ARMS
    env = SimEnv(item.config, item.board, seed=item.seed, run_dir=run_dir, render=model_arm)

    def between(i: int) -> None:
        if item.swap and i == 0:
            env.inject_swap()

    label, provider = "no model", "none"
    with EventWriter(run_dir) as writer:
        if job.arm == "linear":
            summary = LinearAgent(env, item.board, writer, task_budget=job.budget, between_tasks=between).run(TASKS)
        else:
            model: ReasoningModel | None = None
            log = None
            if model_arm:
                model, label = model_for(job.provider, item.seed)
                provider = job.provider
                log = CallLog(run_dir / "model_calls.jsonl")
            agent = Agent(env, item.board, writer, agent_config(job.arm, job.budget),
                          proposer=ModelProposer(model, label, log) if model is not None else None,
                          model_name=getattr(model, "model", "none"), provider=provider,
                          frames_root=run_dir if model_arm else None, prose_model=model, call_log=log,
                          between_tasks=between)
            summary = agent.run(TASKS)
    record = score_run(job.arm, item.name, item.seed, item.trap, item.swap, item.config, summary,
                       env.truth.hazard_ticks, env.truth.min_hazard_distance, env.truth.swapped_at, label, provider)
    if not job.keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return record


def _exit_on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def done_pairs(out: Path) -> set[tuple[str, str]]:
    if not out.exists():
        return set()
    return {(r["arm"], r["item"]) for r in read_records(out)}


def read_records(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_set(
    items: Sequence[bench.Item],
    arms: Iterable[Arm],
    out: Path,
    budget: int,
    provider: str = "fake",
    workers: int = 1,
    runs_root: Path | None = None,
    keep_runs: bool = False,
    on_record: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Run every pending (arm, item); append each record to `out` as it finishes. `on_record`
    returning False stops after the current runs (the live dollar cap). The no-repair arm runs
    only on rewire items (elsewhere it is the model arm)."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS} (memory)")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = runs_root or out.parent / f"{out.stem}_runs"
    skip = done_pairs(out)
    jobs = [Job(arm, it, budget, provider, root, keep_runs) for arm in arms for it in items
            if (arm, it.name) not in skip and (arm != "no_repair" or it.swap)]
    fresh: list[dict[str, Any]] = []
    main = threading.current_thread() is threading.main_thread()
    before = signal.signal(signal.SIGTERM, _exit_on_sigterm) if main else None
    try:
        with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:  # exit terminates
            for record in pool.imap_unordered(run_job, jobs):
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
                fresh.append(record)
                print(f"{record['arm']:>15} {record['item']} {'ok  ' if record['success'] else 'FAIL'} "
                      f"exp {record['experiments']:>3} hazard {record['hazard_ticks']} {record['wall_s']:.0f}s",
                      flush=True)
                if on_record is not None and not on_record(record):
                    pool.terminate()
                    break
    finally:
        if main:
            signal.signal(signal.SIGTERM, before)
    return fresh
