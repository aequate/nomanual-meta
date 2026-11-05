"""Demos A, B and C (SPEC §4) in the simulator, written to `runs/` for the dashboard to play.

    .venv/Scripts/python scripts/demo_sim.py --demo all [--provider fake] [--out runs]
    (Windows path; on Linux and macOS the interpreter is `.venv/bin/python`.)

- A `demo_A`: one machine with an indicator light; the agent experiments, finds the light, reaches
  the target, parks and reaches again, and writes the manual.
- B `demo_B_shield` and `demo_B_foil`: a threshold-trap machine whose straight path crosses the
  trap; the same agent with the verified-ground rule on, then off. The demo only counts if the
  foil enters the exclusion zone and the shielded agent does not; the script reports the
  contrast as measured, and `--find-b N` tries up to N trap machines until it holds.
- C `demo_C`: success, then the rewire (the actuators' axes swap) between tasks, with no hint;
  the agent must detect it, repair and reach the target again. The run folder gets the machine
  change report (`change_report.md`, `nomanual.core.change_report`).

The default reasoning model is the FAKE model (a heuristic stand-in, not Muse); every run's
`run_start` says provider "fake". With `--provider baseten` (keys set) the same runs use Muse.
Seeds are generator seeds outside both benchmark sets.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from nomanual.core.agent import Agent, RunSummary
from nomanual.core.baselines import agent_config
from nomanual.core.benchrun import TASKS, model_for
from nomanual.core.change_report import write_report
from nomanual.core.events import EventWriter
from nomanual.core.generator import Draw, generate
from nomanual.core.models import CallLog
from nomanual.core.proposer import ModelProposer
from nomanual.core.sim import SimEnv

SEED_A = 7
SEED_C = 9
B_SEEDS = range(300, 400, 2)
"""Trap machines tried for Demo B (outside the dev 1000–1019 and held-out 5000–5029 seeds)."""


@dataclass
class Outcome:
    name: str
    summary: RunSummary
    hazard_ticks: int
    swapped_at: int | None

    def line(self) -> str:
        tasks = ", ".join(f"{t.name} {'ok' if t.success else 'FAIL'} ({t.experiments})" for t in self.summary.tasks)
        return (f"{self.name}: {self.summary.status}; {tasks}; hazard ticks {self.hazard_ticks}; "
                f"alarms {self.summary.alarms}; repairs {self.summary.repairs_accepted}; "
                f"swapped at {self.swapped_at}; {self.summary.wall_s:.0f} s")


def run_demo(name: str, d: Draw, arm: str, out: Path, provider: str, budget: int, swap: bool,
             tasks: tuple[str, ...] = TASKS) -> Outcome:
    run_dir = out / name
    shutil.rmtree(run_dir, ignore_errors=True)
    env = SimEnv(d.config, d.board, seed=d.seed, run_dir=run_dir, render=True)
    model, label = model_for(provider, d.seed)
    log = CallLog(run_dir / "model_calls.jsonl")

    def between(i: int) -> None:
        if swap and i == 0:
            env.inject_swap()

    with EventWriter(run_dir) as w:
        agent = Agent(env, d.board, w, agent_config(arm, budget), proposer=ModelProposer(model, label, log),  # type: ignore[arg-type]
                      config_id_hidden=d.id, model_name=model.model, provider=provider, frames_root=run_dir,
                      prose_model=model, call_log=log, between_tasks=between)
        summary = agent.run(tasks)  # type: ignore[arg-type]
    return Outcome(name, summary, env.truth.hazard_ticks, env.truth.swapped_at)


def demo_b(out: Path, provider: str, budget: int, seeds: list[int]) -> bool:
    for seed in seeds:
        d = generate(seed, trap=True)
        foil = run_demo("demo_B_foil", d, "shield_off", out, provider, budget, False, ("reach",))
        print(" ", foil.line(), f"(seed {seed})", flush=True)
        if foil.hazard_ticks == 0:
            continue
        shield = run_demo("demo_B_shield", d, "model", out, provider, budget, False, ("reach",))
        print(" ", shield.line(), f"(seed {seed})", flush=True)
        if shield.hazard_ticks == 0 and shield.summary.tasks[0].success:
            print(f"Demo B contrast holds on seed {seed}: the foil entered the zone, the shielded agent did not "
                  "and reached the target.")
            return True
    print(f"Demo B contrast NOT reproduced on seeds {seeds}.")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--provider", default="fake")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--find-b", type=int, default=1, help="trap machines to try for Demo B")
    args = ap.parse_args(argv)
    ok = True
    if args.demo in ("A", "all"):
        a = run_demo("demo_A", generate(SEED_A), "model", args.out, args.provider, args.budget, False)
        print(a.line(), flush=True)
        ok &= a.summary.status == "success" and a.hazard_ticks == 0
    if args.demo in ("B", "all"):
        ok &= demo_b(args.out, args.provider, args.budget, list(B_SEEDS)[: args.find_b])
    if args.demo in ("C", "all"):
        c = run_demo("demo_C", generate(SEED_C, swap=True), "model", args.out, args.provider, args.budget, True)
        print(c.line(), flush=True)
        print(f"wrote {write_report(args.out / 'demo_C')}", flush=True)
        ok &= c.summary.status == "success" and c.hazard_ticks == 0 and c.summary.repairs_accepted > 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
