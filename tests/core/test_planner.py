import numpy as np

from nomanual.core.board import Board, CircleTarget, RectHazard
from nomanual.core.env import Action
from nomanual.core.events import Plan as PlanEvent
from nomanual.core.planner import MOVE_MIN, Planner, menu, primitives, separation
from nomanual.core.predict import SIM_LIMITS, Belief, Compiled
from nomanual.core.rules import Program
from nomanual.core.shield import Shield

BAR = RectHazard(x0=0.45, y0=0.3, x1=0.55, y1=0.7)
BOARD = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.75, 0.5), r=0.05), hazards=[BAR])
OPEN = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.7, 0.5), r=0.05))
BASE = {
    "actuators": {"A": {"axis": "x", "sign": 1}, "D": {"axis": "y", "sign": 1}},
    "interlock": {"channel": "C", "gates": ["D"]},
    "mode": {"channel": "B"},
}
RIGHT = Program.from_json(BASE)
LEFT = Program.from_json({**BASE, "actuators": {"A": {"axis": "x", "sign": -1}, "D": {"axis": "y", "sign": 1}}})


def belief(*programs: Program) -> Belief:
    comp = Compiled.of(list(programs))
    return Belief(comp, np.full((len(comp), 4), SIM_LIMITS.rate_nominal), np.zeros(4, int), np.ones((len(comp), 2), bool))


def test_menu_and_primitives() -> None:
    m = menu()
    assert len(m) == len({a.short() for a in m}) == 32 + 48 - 12 + 8  # A+1 with C held = C+1 with A held
    assert Action.of(0.2, D=-0.25) in m
    prims = primitives(RIGHT)
    assert Action.of(2.0, D=1, C=1) in prims and Action.of(0.5, A=-1) in prims
    assert Action.of(0.2, B=1) in prims and all(not a.is_neutral() for a in prims)


def test_separation_counts_pairs_whose_predictions_differ() -> None:
    b = belief(RIGHT, LEFT)
    acts = [Action.of(1.0, A=1), Action.of(0.2, B=1), Action.of(1.0, D=1, C=1)]
    pred = b.predict(acts, (0.5, 0.5))[0]
    sep = separation(pred.end, pred.sigma, np.array([0.5, 0.5]))
    assert sep[0] > 0.9 and sep[1] < 0.01 and sep[2] < 0.01
    assert separation(pred.end[:1], pred.sigma[:1], np.array([1.0])).tolist() == [0.0, 0.0, 0.0]
    # The light shows the mode under one hypothesis only: a press separates them when tracked.
    lit = Program.from_json({**BASE, "indicator": {"observable": "light_1", "shows": "mode"}}, ("light_1",))
    b2 = belief(RIGHT, lit)
    p2 = b2.predict(acts, (0.5, 0.5))[0]
    assert separation(p2.end, p2.sigma, np.array([0.5, 0.5]), p2.mode, b2.comp.indicator)[1] == 1.0
    # Without the rule the light stays as it is now (as the verifier scores it), not off: with the
    # light on, the press that lights it under `lit` separates nothing.
    on = separation(p2.end, p2.sigma, np.array([0.5, 0.5]), p2.mode, b2.comp.indicator, light_now=1)
    assert on[1] < 0.01 and p2.mode[1, 1] == 1


def test_disagreeing_survivors_get_a_separating_experiment() -> None:
    sh = Shield(OPEN)
    plan = Planner(OPEN, sh).choose(belief(RIGHT, LEFT), (0.5, 0.5), 0.003, OPEN.target.center, 0.04)
    assert plan.action is not None and "A" in plan.action.active()
    chosen = plan.candidates[plan.chosen]
    assert chosen.separation > 0.9 and "disagree" in plan.why
    PlanEvent.model_validate(plan.event())


def test_one_survivor_follows_a_route_around_the_hazard() -> None:
    sh = Shield(BOARD)
    pl = Planner(BOARD, sh)
    b = belief(RIGHT)
    route = pl.route(b, (0.3, 0.5), 0.003, BOARD.target.center, 0.04)
    assert route is not None and len(route.actions) >= 4
    pts = np.array(route.points)
    assert np.linalg.norm(pts[-1] - np.array(BOARD.target.center)) <= 0.04
    assert (BOARD.hazard_distance(pts) >= sh.config.margin).all()
    assert pts[:, 1].max() > BAR.y1 or pts[:, 1].min() < BAR.y0  # it goes round the bar
    plan = pl.choose(b, (0.3, 0.5), 0.003, BOARD.target.center, 0.04)
    assert plan.action is not None and "route step" in plan.why
    assert plan.route is not None and plan.action.active() == plan.route.actions[0].active()


def test_no_safe_candidate_means_no_action() -> None:
    tight = Board(start=(0.5, 0.5), target=CircleTarget(center=(0.7, 0.7), r=0.05),
                  hazards=[RectHazard(x0=0.0, y0=0.0, x1=1.0, y1=0.47), RectHazard(x0=0.0, y0=0.53, x1=1.0, y1=1.0)])
    plan = Planner(tight, Shield(tight)).choose(belief(), (0.5, 0.5), 0.003, (0.7, 0.7), 0.04)
    assert plan.action is None and plan.chosen is None and "no candidate" in plan.why


def test_vetoes_count_the_candidates_the_shield_removes() -> None:
    planner = Planner(BOARD, Shield(BOARD))
    acts = [Action.of(2.0, A=1), Action.of(0.2, B=1), Action.of(0.2, A=-0.25)]
    # From just left of the bar, two seconds of A+ would enter it.
    scored = planner.score(acts, belief(RIGHT), (0.42, 0.5), 0.003, BOARD.target.center)
    kept = [c.action for c in scored]
    assert Action.of(2.0, A=1) not in kept
    assert planner.shield.vetoes == len(acts) - len(kept) >= 1
    assert planner.shield.interventions == 0  # planning does not count as an intervention


def test_stuck_with_no_room_to_gain_explores_an_untried_move() -> None:
    # No route (the goal can never be reached), nothing to separate, everything measured here
    # but one move, and no hazards to make room from: the agent explores the untried move
    # instead of repeating a probe that shows nothing.
    planner = Planner(OPEN, Shield(OPEN))
    pos = (0.5, 0.5)
    fresh = Action.of(0.5, A=-1)
    tried = [(a, pos) for a in planner.menu if a != fresh]
    plan = planner.choose(belief(RIGHT), pos, 0.003, pos, -1.0, tried=tried)
    assert plan.route is None and plan.action == fresh
    chosen = plan.candidates[plan.chosen]
    assert chosen.novel and chosen.move >= MOVE_MIN and "exploring a move not tried here" in plan.why


def test_exploring_never_prefers_an_untried_tap_that_moves_nothing() -> None:
    # Every move was tried here except a mode tap the leader says moves nothing: exploring takes
    # the longest predicted move instead, which gets the tip somewhere new.
    planner = Planner(OPEN, Shield(OPEN))
    pos = (0.5, 0.5)
    tried = [(a, pos) for a in planner.menu if a.short() != Action.of(0.2, B=1).short()]
    plan = planner.choose(belief(RIGHT), pos, 0.003, pos, -1.0, tried=tried)
    chosen = plan.candidates[plan.chosen]
    every = planner.score(planner.menu, belief(RIGHT), pos, 0.003, pos, tried=tried)
    assert plan.route is None and "exploring a move tried before" in plan.why
    assert not chosen.novel and round(chosen.move, 3) == round(max(c.move for c in every), 3) >= MOVE_MIN


def test_exploring_heads_away_from_recent_starts_instead_of_rocking() -> None:
    # Everything was tried here, and the tip was just at the end of the long move right: exploring
    # takes the long move left, not back to where the tip just came from.
    planner = Planner(OPEN, Shield(OPEN))
    pos = (0.5, 0.5)
    tried = [(a, pos) for a in planner.menu] + [(Action.of(2.0, A=-1), (0.572, 0.5))]
    plan = planner.choose(belief(RIGHT), pos, 0.003, pos, -1.0, tried=tried)
    assert plan.route is None and plan.action is not None and plan.action.channels["A"] < 0
    assert "exploring a move tried before" in plan.why


def test_the_routes_first_step_passes_the_shield_against_every_survivor() -> None:
    # The route is screened against the heaviest hypotheses only. A light survivor says A moves
    # the tip down, toward the hazard below, within reach of an unknown drop of it: the shield
    # refuses every A move here, even the shortest, so a route starting with one leads nowhere
    # (on enumeration seed 2 the agent backed off and came back to such a route in a loop until
    # the budget ran out). With every survivor as the guard, the route starts another way.
    below = RectHazard(x0=0.1, y0=0.0, x1=0.9, y1=0.38)
    board = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.7, 0.5), r=0.05), hazards=[below])
    down = Program.from_json({**BASE, "actuators": {"A": {"axis": "y", "sign": -1}, "D": {"axis": "y", "sign": 1}}})
    pl = Planner(board, Shield(board))
    lead, guard = belief(RIGHT), belief(RIGHT, down)
    pos, sigma, goal = (0.3, 0.5), 0.001, board.target.center
    first = pl.route(lead, pos, sigma, goal, 0.04)
    assert first is not None and "A" in first.actions[0].active()
    assert pl.shield.decide(first.actions[0], pos, sigma, guard).final is None
    guarded = pl.route(lead, pos, sigma, goal, 0.04, guard=guard)
    assert guarded is not None and "A" not in guarded.actions[0].active()
    assert pl.shield.decide(guarded.actions[0], pos, sigma, guard).final is not None


def test_a_mode_tap_is_no_way_round_a_first_move_the_shield_refuses() -> None:
    # Hazards above and below, and a survivor saying A moves the tip down: every move here is
    # refused against every survivor and only the mode tap passes. A route "tap, then A" would
    # be tapped again at every plan (live Demo C #2: about 80 `D+1 0.2s` in a row, no motion, no
    # alarm). There is no route then, and the stuck logic picks a move that goes somewhere.
    walls = [RectHazard(x0=0.1, y0=0.0, x1=0.9, y1=0.38), RectHazard(x0=0.1, y0=0.62, x1=0.9, y1=1.0)]
    board = Board(start=(0.3, 0.5), target=CircleTarget(center=(0.7, 0.5), r=0.05), hazards=walls)
    down = Program.from_json({**BASE, "actuators": {"A": {"axis": "y", "sign": -1}, "D": {"axis": "y", "sign": 1}}})
    pl = Planner(board, Shield(board))
    lead, guard = belief(RIGHT), belief(RIGHT, down)
    pos, sigma, goal = (0.3, 0.5), 0.001, board.target.center
    prims = primitives(RIGHT)
    assert [a for a, ok in zip(prims, pl._runnable(prims, pos, sigma, guard), strict=True) if ok] == [Action.of(0.2, B=1)]
    assert pl.route(lead, pos, sigma, goal, 0.04, guard=guard) is None
    plan = pl.choose(lead, pos, sigma, goal, 0.04, guard=guard)
    assert plan.action is not None and plan.action.active() != ("B",)
    assert plan.candidates[plan.chosen].move > 0.0


def test_loop_detection_distinguishes_oscillation_from_detours_and_tracking_noise():
    from nomanual.core.planner import _looped
    action = Action.of(.5, A=1)
    assert _looped((.5, .5), [(action, (.5 if i % 2 == 0 else .482, .5)) for i in range(12)])
    assert not _looped((.38, .5), [(action, (.5-i*.01, .5)) for i in range(12)])
    assert not _looped((.5, .5), [(action, (.5 if i % 2 == 0 else .501, .5)) for i in range(12)])
    assert not _looped((.5, .5), [(action, (.5 if i % 2 == 0 else .482, .5)) for i in range(6)])


def test_repeated_route_prefix_yields_to_a_forward_move_that_the_real_shield_accepts(monkeypatch):
    from nomanual.core.planner import Route
    planner = Planner(OPEN, Shield(OPEN))
    pos = (.5, .5); backward = Action.of(.5, A=-1)
    # Isolate the chooser from search: a supplied route prefix starts by backing off.
    monkeypatch.setattr(planner, 'route', lambda *a, **kw: Route([backward], [pos, (.482, .5)]))
    b = belief(RIGHT)
    assert planner.choose(b, pos, .003, OPEN.target.center, .04).action == backward
    tried = [(Action.of(.5, A=-1 if i % 2 == 0 else 1),
              (.5 if i % 2 == 0 else .482, .5)) for i in range(12)]
    plan = planner.choose(b, pos, .003, OPEN.target.center, .04, tried=tried)
    assert plan.action.channels['A'] > 0 and 'route repeats' in plan.why
    assert planner.shield.screen([plan.action], pos, .003, b)[0][0]


def test_forward_escape_can_take_a_probe_shorter_than_the_exploration_move_floor(monkeypatch):
    from nomanual.core.planner import Route
    wall = Board(start=(.5, .5), target=CircleTarget(center=(.8, .5), r=.04),
                 hazards=[RectHazard(x0=.65, y0=.3, x1=.75, y1=.7)])
    planner = Planner(wall, Shield(wall)); pos = (.5, .5); b = belief(RIGHT)
    backward = Action.of(.5, A=-1)
    monkeypatch.setattr(planner, 'route', lambda *a, **kw: Route([backward], [pos, (.482, .5)]))
    tried = [(Action.of(.5, A=-1 if i % 2 == 0 else 1),
              (.5 if i % 2 == 0 else .482, .5)) for i in range(12)]
    plan = planner.choose(b, pos, .001, wall.target.center, .04, tried=tried)
    chosen = plan.candidates[plan.chosen]
    assert 'route repeats' in plan.why and 0 < chosen.move < MOVE_MIN
    assert chosen.progress > 0 and planner.shield.screen([plan.action], pos, .001, b)[0][0]
