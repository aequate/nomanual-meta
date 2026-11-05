"""One real hypothesis request per configured provider (the human runs this once keys are set).

    .venv/Scripts/python scripts/live_smoke.py [--providers baseten meta] [--fake]
    (Windows path; on Linux and macOS the interpreter is `.venv/bin/python`.)

For each provider whose environment variables are set (see `models.PROVIDERS`), it sends the
hypothesis prompt for a short probe record from dev machine 0 and prints latency, tokens, cost
(only when `NOMANUAL_PRICE_<PROVIDER>` is set) and whether the reply validated as programs in
the rule language. A provider without its variables is reported as SKIP with the names it needs;
nothing is guessed. `--fake` runs the same path on the fake model (not Muse) to check the script.
Calls are logged to `runs/live_smoke/model_calls.jsonl`. Exit code 1 when a configured provider
failed, else 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nomanual.core import bench
from nomanual.core.agent import Context
from nomanual.core.env import CHANNELS, Action, StepResult
from nomanual.core.models import PROVIDERS, CallLog, FakeModel, ModelError, ReasoningModel, make_model, missing_env
from nomanual.core.proposer import FAKE_LABEL, ModelProposer, fake_responder
from nomanual.core.sim import SimEnv


def probe_context() -> Context:
    """Each control alone, both signs, then each paired with every other held: 32 experiments."""
    item = bench.load("dev")[0]
    env = SimEnv(item.config, item.board, seed=item.seed)
    env.reset()
    results: list[StepResult] = []
    for ch in CHANNELS:
        for s in (1.0, -1.0):
            results.append(env.step(Action.of(0.5, **{ch: s})))
            for held in CHANNELS:
                if held != ch:
                    results.append(env.step(Action.of(0.5, **{ch: s, held: 1.0})))
    return Context(results=results, seqs=list(range(len(results))), observables=(), survivors=[], leader=None,
                   misfits=[], pos=results[-1].after.pos, board=item.board)


def smoke(name: str, model: ReasoningModel, label: str, ctx: Context, log: CallLog) -> bool:
    got = ModelProposer(model, label, log).hypotheses(ctx)
    u = got.usage
    cost = ("no cost (fake)" if name == "fake" else f"${u.cost_usd:.5f}" if u.cost_usd > 0
            else "cost unknown (no NOMANUAL_PRICE_ set)")
    ok = bool(got.programs) and not got.error
    print(f"{name:>10}: {'OK  ' if ok else 'FAIL'} latency {u.latency_s:.2f} s, tokens in {u.tokens_in} out "
          f"{u.tokens_out}, {cost}; {len(got.programs)} valid program(s)"
          + (f"; error: {got.error[:200]}" if got.error else ""))
    if got.programs:
        print(f"{'':>12}first: {got.programs[0].text()}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS), default=sorted(PROVIDERS))
    ap.add_argument("--fake", action="store_true", help="also run the fake model (not Muse)")
    ap.add_argument("--log-dir", type=Path, default=Path("runs/live_smoke"))
    args = ap.parse_args(argv)
    ctx = probe_context()
    log = CallLog(args.log_dir / "model_calls.jsonl")
    failed = False
    if args.fake:
        smoke("fake", FakeModel(fake_responder()), FAKE_LABEL, ctx, log)
    for name in args.providers:
        missing = missing_env(name)
        if missing:
            print(f"{name:>10}: SKIP (set {', '.join(missing)})")
            continue
        try:
            model = make_model(name)
        except ModelError as e:
            print(f"{name:>10}: FAIL {e}")
            failed = True
            continue
        if not PROVIDERS[name].verified:
            print(f"{name:>10}: note: endpoint/model id not verified ({PROVIDERS[name].note})")
        failed |= not smoke(name, model, f"{name} {model.model}", ctx, log)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
