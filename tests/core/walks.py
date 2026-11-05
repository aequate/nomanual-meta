"""Data for verifier and search tests: simulated experiments that stay away from the soft limits."""

import random

from nomanual.core.env import Action, StepResult
from nomanual.core.generator import Draw
from nomanual.core.sim import SimEnv


def interior_walk(d: Draw, seed: int, n: int, swap_at: int | None = None) -> list[StepResult]:
    """Random full-scale actions; near the edge, the move whose true outcome (tried on a copy of
    the hidden machine: test only) heads back to the middle. `swap_at`: rewire before that step."""
    env = SimEnv(d.config, d.board, seed=seed, swap_before_step=swap_at)
    env.reset()
    rng = random.Random(seed)
    cfg = d.config
    out = []
    moves = [{c: s, cfg.interlock.channel: 1.0} for c in cfg.actuators for s in (-1.0, 1.0)]
    for _ in range(n):
        ch = {c: rng.choice([-1.0, 0.0, 0.0, 1.0]) for c in "ABCD"}
        x, y = env.true_board_pos()
        if not (0.3 < x < 0.7 and 0.3 < y < 0.7):
            snap = env.machine.snapshot()

            def dist_after(m: dict) -> float:
                env.machine.run(m, 1.0)
                q = env.true_board_pos()
                env.machine.restore(snap)
                return (q[0] - 0.5) ** 2 + (q[1] - 0.5) ** 2

            ch = min(moves, key=dist_after)
        out.append(env.step(Action(channels=ch, duration=rng.choice([0.2, 0.5, 1.0, 2.0]))))
    return out
