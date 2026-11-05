"""Route preferences account for spread while the independent shield keeps every survivor."""
from types import SimpleNamespace

import numpy as np

from nomanual.core.agent import Agent, AgentConfig, _route_costs
from nomanual.core.env import Action
from nomanual.core.predict import Batch


def test_vague_prediction_does_not_win_just_by_dividing_residuals():
    # Three 2D observations: the broad explanation has lower z² but doubles sigma.
    costs = np.array([12.0, 7.0])
    sigma = np.array([[.01] * 3, [.02] * 3])
    got = _route_costs(costs, sigma, np.ones(3, bool))
    assert got[0] < got[1]
    # Better agreement can still justify the wider prediction; this is not a rule ban.
    assert _route_costs(np.array([100.0, 0.0]), sigma, np.ones(3, bool))[1] < 100
    assert np.array_equal(costs, [12, 7])


def test_unusable_data_and_a_common_sigma_scale_supply_no_preference():
    costs = np.array([12.0, 7.0])
    sigma = np.array([[.01, .01], [.02, .5]])
    usable = np.array([True, False])
    got = _route_costs(costs, sigma, usable)
    assert np.allclose(got, _route_costs(costs, sigma * 10, usable))
    assert np.array_equal(costs, _route_costs(costs, sigma, np.zeros(2, bool)))


def test_changed_model_route_weights_leave_guard_order_and_membership_unchanged():
    scores = SimpleNamespace(alive=np.array([True, True]),
                             sigma=np.array([[.01] * 3, [.02] * 3]))
    pool = SimpleNamespace(scores=scores, costs=lambda: np.array([12.0, 7.0]))
    agent = Agent.__new__(Agent)
    agent.seg = SimpleNamespace(pool=pool)
    batch = Batch.candidates([Action.of(.2, A=1)] * 3, (.3, .2), np.zeros(4, int))
    batch.w = np.ones(3)
    agent._batch = lambda: batch
    agent.config = AgentConfig(hypotheses='enum')
    old_plan, _, old_guard = agent._ranked()
    agent.config = AgentConfig(hypotheses='model')
    plan, weights, guard = agent._ranked()
    assert old_plan.tolist() == old_guard.tolist() == [1, 0]
    assert plan.tolist() == [0, 1] and weights[0] > weights[1]
    assert np.array_equal(guard, old_guard)
    assert scores.alive.all() and np.array_equal(scores.sigma[1], [.02] * 3)
