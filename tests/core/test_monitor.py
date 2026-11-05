import numpy as np
import pytest

from nomanual.core.generator import generate
from nomanual.core.monitor import H_ALARM, Monitor, edits, rewired, suspects
from nomanual.core.predict import Batch, Compiled
from nomanual.core.rules import Program, program_from_config
from nomanual.core.verify import score
from tests.core.walks import interior_walk


def feed(m: Monitor, zs: list[float]) -> int | None:
    for i, z in enumerate(zs):
        if m.update(z, i) is not None:
            return i
    return None


def test_cusum_ignores_noise_and_single_outliers() -> None:
    rng = np.random.default_rng(0)
    assert feed(Monitor(), list(np.abs(rng.normal(0, 1, 500)))) is None
    assert feed(Monitor(), [1.0] * 10 + [50.0] + [1.0] * 10) is None
    assert feed(Monitor(), [1.0, 1.0, 5.0, 1.0, 5.0, 1.0, 1.0, 5.0, 1.0]) is None  # scattered misses


def test_cusum_fires_on_a_persistent_mismatch() -> None:
    m = Monitor()
    at = feed(m, [1.0] * 5 + [5.0] * 10)
    assert at is not None and 5 + 2 <= at <= 5 + 4
    alarm = Monitor().update(5.0, 0)
    assert alarm is None
    m = Monitor()
    for i in range(3):
        m.update(6.0, 10 + i)
    a = m.update(6.0, 13)
    assert a is not None and a.window == [10, 11, 12, 13] and a.stat > H_ALARM
    assert set(a.event()) == {"stat", "threshold", "suspect_rules", "cause"} and a.cause == "drift"
    before = m.stat
    assert m.update(float("nan"), 14) is None and m.stat == before  # unusable: nothing changes
    m.reset()
    assert m.stat == 0 and not m.window


def test_rewired_is_an_involution_and_an_edit() -> None:
    p = Program.from_json({
        "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": -1}},
        "interlock": {"channel": "C", "gates": ["D"]},
        "mode": {"channel": "B"},
        "thresholds": [{"axis": "x", "op": ">", "value": 0.6, "channel": "D", "effect": "drop",
                        "drop_axis": "x", "drop_dir": -1}],
    })
    r = rewired(p)
    assert r.act["A"].axis == "y" and r.act["D"].axis == "x"
    assert r.thresholds[0].axis == "y" and r.thresholds[0].drop_axis == "y"
    assert rewired(r) == p
    assert edits(p)[0] == r and p not in edits(p)


@pytest.mark.parametrize("seed", [5])
def test_swap_raises_an_alarm_and_blames_the_actuators(seed: int) -> None:
    d = generate(seed, trap=False)
    truth = program_from_config(d.config).replace(indicator=None)
    res = interior_walk(d, seed, 34, swap_at=25)
    m = Monitor()
    alarm = None
    for i in range(len(res)):
        # As the agent sees it: the newest experiment's residual under rates fitted so far.
        z = score(Compiled.of([truth]), Batch.of_results(res[: i + 1])).z[0, i]
        alarm = m.update(float(z), i)
        if alarm is not None:
            break
    assert alarm is not None and 25 <= alarm.window[0] and i <= 25 + 6
    window = Batch.of_results([res[k] for k in alarm.window], seqs=alarm.window)
    sus, best = suspects(truth, window)
    acts = {r.text for r in truth.rules() if r.kind == "actuator"}
    assert sus and sus[0] in acts
    assert best == rewired(truth)
