"""Shared builders and Hypothesis strategies for machine configurations."""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

from nomanual.core.env import CHANNELS
from nomanual.core.machine import MachineConfig


def make_config(**overrides: Any) -> MachineConfig:
    """The CONTRACTS §3.2 example, with optional overrides of top-level keys."""
    base: dict[str, Any] = {
        "id": "m0042",
        "roles": {"A": "actuator", "B": "mode", "C": "interlock", "D": "actuator"},
        "actuators": {
            "A": {"axis": "u", "sign": 1, "gain": 1.0},
            "D": {"axis": "v", "sign": -1, "gain": 0.8},
        },
        "interlock": {"channel": "C", "gates": ["A"], "threshold": 0.3},
        "mode": {"channel": "B", "threshold": 0.3, "initial": 0},
        "conditional": [],
        "thresholds": [],
        "indicator": {"present": False, "shows": "none"},
        "start": [0.3, 0.2],
    }
    base.update(overrides)
    return MachineConfig.model_validate(base)


@st.composite
def effects(draw: st.DrawFn, kinds: tuple[str, ...]) -> dict[str, Any]:
    kind = draw(st.sampled_from(kinds))
    if kind == "gain_mult":
        return {"gain_mult": draw(st.floats(0.25, 2.0))}
    if kind == "drop":
        return {"drop": {"axis": draw(st.sampled_from(["u", "v"])), "amount": draw(st.floats(-0.1, 0.1))}}
    return {kind: True}


@st.composite
def configs(draw: st.DrawFn) -> MachineConfig:
    order = draw(st.permutations(list(CHANNELS)))
    acts, il, mode = order[:2], order[2], order[3]
    roles = {acts[0]: "actuator", acts[1]: "actuator", il: "interlock", mode: "mode"}
    actuators = {
        ch: {
            "axis": draw(st.sampled_from(["u", "v"])),
            "sign": draw(st.sampled_from([1, -1])),
            "gain": draw(st.floats(0.5, 1.5)),
        }
        for ch in acts
    }
    gates = draw(st.sampled_from([[acts[0]], [acts[1]], list(acts)]))
    conditional = [
        {"channel": draw(st.sampled_from(acts)), "mode": draw(st.sampled_from([0, 1])),
         "effect": draw(effects(("gain_mult", "sign_flip", "axis_swap")))}
        for _ in range(draw(st.integers(0, 2)))
    ]
    thresholds = [
        {
            "axis": draw(st.sampled_from(["u", "v"])),
            "op": draw(st.sampled_from([">", "<"])),
            "value": draw(st.floats(0.1, 0.9)),
            "channel": draw(st.sampled_from(acts)),
            "effect": draw(effects(("gain_mult", "sign_flip", "stop", "drop"))),
        }
        for _ in range(draw(st.integers(0, 2)))
    ]
    return MachineConfig.model_validate(
        {
            "id": "prop",
            "roles": roles,
            "actuators": actuators,
            "interlock": {"channel": il, "gates": gates, "threshold": draw(st.floats(0.1, 0.9))},
            "mode": {"channel": mode, "threshold": draw(st.floats(0.1, 0.9)), "initial": draw(st.sampled_from([0, 1]))},
            "conditional": conditional,
            "thresholds": thresholds,
            "indicator": {"present": draw(st.booleans()), "shows": draw(st.sampled_from(["mode", "none"]))},
            "start": [draw(st.floats(0.05, 0.95)), draw(st.floats(0.05, 0.95))],
        }
    )


channel_values = st.sampled_from([-1.0, -0.6, -0.3, -0.04, 0.0, 0.04, 0.3, 0.6, 1.0]) | st.floats(-1.0, 1.0)
actions = st.tuples(
    st.fixed_dictionaries({ch: channel_values for ch in CHANNELS}),
    st.floats(0.2, 2.0),
)
