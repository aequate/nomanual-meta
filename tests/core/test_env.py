import pytest
from pydantic import ValidationError

from nomanual.core.env import CHANNELS, Action, Env, Observation


def test_action_fills_missing_channels_with_zero() -> None:
    a = Action(channels={"A": 1.0}, duration=1.0)
    assert a.channels == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    assert tuple(a.channels) == CHANNELS


@pytest.mark.parametrize(
    "channels,duration",
    [
        ({"E": 1.0}, 1.0),
        ({"A": 1.5}, 1.0),
        ({"A": -1.01}, 1.0),
        ({"A": "1"}, 1.0),
        ({"A": True}, 1.0),
        ({"A": 1.0}, 0.1),
        ({"A": 1.0}, 2.5),
    ],
)
def test_action_rejects_out_of_contract(channels: dict, duration: float) -> None:
    with pytest.raises(ValidationError):
        Action(channels=channels, duration=duration)


def test_deadband_and_active() -> None:
    a = Action.of(0.5, A=0.04, B=-0.049, C=0.05, D=-1)
    assert a.effective() == {"A": 0.0, "B": 0.0, "C": 0.05, "D": -1.0}
    assert a.active() == ("C", "D")
    assert Action.neutral().is_neutral()
    assert a.short() == "C+0.05 D-1 0.50s"
    assert Action.of(1.0, A=0.6).short() == "A+0.6 1.00s"


def test_action_json_round_trips() -> None:
    a = Action.of(1.0, A=1, C=-1)
    assert Action.model_validate_json(a.model_dump_json()) == a


def test_observation_defaults() -> None:
    o = Observation(t=0.0, pos=(0.3, 0.2), sigma=0.002, confidence=1.0)
    assert o.extras == {} and o.frame is None


def test_env_protocol_is_structural() -> None:
    class Dummy:
        def reset(self): ...
        def observe(self): ...
        def step(self, action): ...
        def stop(self): ...
        def track(self, name, description): ...

    assert isinstance(Dummy(), Env)
