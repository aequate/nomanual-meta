"""The deterministic planner (SPEC §3.4): which experiment next, and the route to a goal.

**Experiments.** Every candidate action (a fixed menu plus any the model proposes) is predicted
under every surviving hypothesis. Its value is the posterior mass of hypothesis pairs it
separates: a pair counts as separated when the predicted end points lie `SEP_SCALE` combined
standard deviations apart (or the tracked light would differ), smoothly below that. The value
plus a measuring term: an action whose outcome some survivor predicts only loosely (a long
move, a drop, an end next to a drop line) can show a survivor wrong in a way separation by
distance misses, which matters when the truth is not among the survivors yet. It counts only
for an action not yet tried from about here (`NOVEL_R`): repeating a measurement whose outcome
was already as predicted shows nothing new, and without that rule the agent can loop on it. The value is then discounted by
distance from the goal (moves that end farther from the goal lose value), by predicted
uncertainty, by duration, and by a low predicted hazard clearance. Only actions the
shield accepts as they stand are candidates, so the planner never counts on a shrink.

**Route.** A* over the leading hypothesis' predictions from the current state (position and mode
presses) with motion primitives of that hypothesis, where every step must pass the shield's
screen from the predicted position. The first step must also pass the shield as the agent will
check it (against every survivor, shortened if need be): a route whose first step the shield
refuses leads nowhere, and backing off from it only to return makes a loop. The same holds for
mode taps before the first move: a tap then a refused move would make the agent tap again and
again, with no motion for the change monitor to learn from. The agent re-plans after every
step, so the route only has to be right about its first step; the rest is shown on the
knowledge map.

**Choice.** While some candidate separates at least `SEP_MIN` of the hypothesis pairs, the
highest-value candidate runs (the route's first step is one of them, valued for its progress).
Once no candidate separates them, the route's first step runs; with no route, the best move,
unless no move is worth anything (nothing to separate, measure or gain): then the move that
gives the tip the most room from hazards, since the shield lets more actions run there. When no
move gains room either, the agent would only repeat a probe that shows nothing (a mode tap, again
and again), so it explores instead: a move not yet tried from about here (at least `MOVE_MIN` long), which
adds an outcome the survivors have not been checked on; with none, any move. Among those it takes
the one predicted to end farthest from here and from every recent start, so the tip goes
somewhere new instead of rocking back and forth.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nomanual.core.board import Board
from nomanual.core.env import CHANNELS, DURATION_MIN, Action
from nomanual.core.predict import Batch, Belief
from nomanual.core.rules import LEVEL_ON, Program
from nomanual.core.shield import GroundRule, Shield

SEP_SCALE = 4.0
"""Predictions this many combined sigmas apart count as separated (63%); at 5 sigmas, about the
single-experiment kill threshold, 91%."""
SEP_MIN = 0.05
"""Posterior pair mass an experiment must separate to be worth running instead of the route."""
OBS_SIGMA = 0.003
"""Measurement noise added to both predictions when comparing them."""
GOAL_SCALE = 0.15
"""Moving this much farther from the goal costs a factor e of the experiment's value."""
COST_PER_S = 0.001
CLEAR_WANT = 0.08
"""Predicted clearance below this is penalised (the shield enforces the hard margin)."""
CLEAR_PENALTY = 1.0
PROGRESS_WEIGHT = 0.5
"""Value of moving the whole leader-predicted distance to the goal, per unit board distance."""
MENU_DURATIONS = (0.2, 0.5, 1.0, 2.0)
HELD_DURATIONS = (0.5, 2.0)
ROUTE_DURATIONS = (0.5, 1.0, 2.0)
ROUTE_CELL = 0.02
ROUTE_EXPANSIONS = 300
ROUTE_WEIGHT = 3.0
ROUTE_K = 4
"""Hypotheses the route's steps are screened against (the heaviest first)."""
NEAR_COST = 4.0
ROUTE_MIN_STEP = 0.01
"""Greedy A* (weighted heuristic): a short route found fast beats the shortest found late."""
TOP_CANDIDATES = 5
MEASURE_SCALE = 0.03
"""A survivor predicting an outcome this loosely (1 sigma) makes it worth measuring (63%)."""
MEASURE_WEIGHT = 0.3
NOVEL_R = 0.03
"""An action tried from within this distance is not new there (no measuring value)."""
GENTLE = 0.25
"""Input level of the menu's gentle moves: slow enough to pass the hypothesis-free shield test
nearer to a hazard than a full-speed move (shield module docstring)."""
STUCK_VALUE = 0.01
"""Below this best value with no route, the planner makes room instead (module docstring)."""
ROOM_MIN = 0.005
"""Less hazard room than this gained is no room: the planner explores instead."""
MOVE_MIN = 0.01
"""A move the leader predicts shorter than this shows nothing new when exploring."""


def menu() -> list[Action]:
    """Hypothesis-free experiments: each channel alone both ways at four durations, each channel
    both ways with another channel held, at two durations, and each channel both ways gently."""
    out = [Action.of(d, **{ch: s}) for ch in CHANNELS for s in (1.0, -1.0) for d in MENU_DURATIONS]
    out += [Action.of(MENU_DURATIONS[0], **{ch: s * GENTLE}) for ch in CHANNELS for s in (1.0, -1.0)]
    for ch in CHANNELS:
        for held in CHANNELS:
            if held != ch:
                out += [Action.of(d, **{ch: s, held: 1.0}) for s in (1.0, -1.0) for d in HELD_DURATIONS]
    unique = {a.short(): a for a in out}
    return list(unique.values())


def primitives(p: Program) -> list[Action]:
    """Motion under hypothesis `p`: each actuator both ways (interlock held when gated), and a
    mode press."""
    out = []
    for ch, _ in p.actuators:
        chans = {ch: 1.0}
        if ch in p.gates:
            chans[p.interlock] = 1.0
        for s in (1.0, -1.0):
            out += [Action.of(d, **{**chans, ch: s}) for d in ROUTE_DURATIONS]
    out.append(Action.of(0.2, **{p.mode: 1.0}))
    return out


@dataclass
class Candidate:
    action: Action
    value: float
    separation: float
    progress: float
    clearance: float
    why: str
    room: float = 0.0
    """Hazard distance of the leader-predicted end point minus that of the current position."""
    move: float = 0.0
    """Leader-predicted distance the tip moves."""
    novel: bool = True
    """Not tried from about here (`NOVEL_R`) among the recent experiments."""
    away: float = 0.0
    """Distance from the leader-predicted end point to here and to every recent start."""

    def event(self) -> dict[str, object]:
        return {"action": self.action.model_dump(), "value": round(self.value, 4), "why": self.why}


@dataclass
class Route:
    actions: list[Action]
    points: list[tuple[float, float]]
    """Leader-predicted positions, starting at the current one."""


@dataclass
class Plan:
    candidates: list[Candidate]
    chosen: int | None
    why: str
    route: Route | None = None
    still: bool = False
    """No candidate (all pass the shield) is predicted to move the tip `MOVE_MIN`: the planner can
    only tap buttons from here."""

    @property
    def action(self) -> Action | None:
        return None if self.chosen is None else self.candidates[self.chosen].action

    def event(self) -> dict[str, object]:
        """`plan` event data (CONTRACTS §6)."""
        return {"candidates": [c.event() for c in self.candidates], "chosen": self.chosen, "why": self.why}


@dataclass
class Planner:
    board: Board
    shield: Shield
    menu: list[Action] = field(default_factory=menu)
    _geo: dict[tuple[float, float, float], np.ndarray] = field(default_factory=dict, repr=False)

    def score(
        self,
        actions: list[Action],
        belief: Belief,
        pos: tuple[float, float],
        sigma: float,
        goal: tuple[float, float],
        weights: np.ndarray | None = None,
        lights: bool = False,
        guard: Belief | None = None,
        tried: Sequence[tuple[Action, tuple[float, float]]] = (),
        light_now: int = 0,
    ) -> list[Candidate]:
        """Value of each action that the shield accepts as it stands (others are left out).
        `guard` is the belief the shield screens with (default `belief`): every survivor, while
        `belief` may be the few the planner weighs. `tried`: (action, start) of recent experiments.
        `light_now`: the tracked light's current reading (what programs without the indicator rule
        predict it stays at)."""
        verdicts = self.shield.screen(actions, pos, sigma, guard if guard is not None else belief)
        safe = [a for a, (ok, _) in zip(actions, verdicts, strict=True) if ok]
        self.shield.vetoes += len(actions) - len(safe)
        if not safe or not len(belief):
            return []
        w = _normalise(weights, len(belief))
        pred = belief.predict(safe, pos, self.shield.config.limits, want_path=True)[0]
        sep = separation(pred.end, pred.sigma, w, pred.mode if lights else None, belief.comp.indicator, light_now)
        lead = int(np.argmax(w))
        end = pred.end[lead]  # (E,2)
        d_now = math.dist(pos, goal)
        d_end = np.linalg.norm(end - np.asarray(goal)[None], axis=1)
        progress = d_now - d_end
        assert pred.path is not None
        pts = pred.path.reshape(-1, 2)
        clear = self.board.hazard_distance(pts).reshape(pred.path.shape[:3]).min(axis=2)  # (H,E)
        clear = (clear - 3 * pred.sigma).min(axis=0)
        unc = pred.sigma[lead]
        measure = w @ (1.0 - np.exp(-((pred.sigma / MEASURE_SCALE) ** 2)))  # (E,)
        done = {a.short() for a, p in tried if math.dist(p, pos) < NOVEL_R}
        novel = np.array([a.short() not in done for a in safe])
        measure *= novel
        move = np.linalg.norm(end - np.asarray(pos)[None], axis=1)
        # How far the predicted end lands from here and from every recent start.
        seen = np.array([pos, *(p for _, p in tried)], dtype=float)
        away = np.linalg.norm(end[:, None, :] - seen[None], axis=2).min(axis=1)
        # Capped at 1 board unit: with no hazards both distances are infinite.
        room = np.minimum(self.board.hazard_distance(end), 1.0) - min(float(self.board.hazard_distance(pos)[0]), 1.0)
        out = []
        for e, a in enumerate(safe):
            discount = math.exp(-max(0.0, -progress[e]) / GOAL_SCALE)
            value = (
                (sep[e] + MEASURE_WEIGHT * measure[e]) * discount
                + PROGRESS_WEIGHT * max(0.0, progress[e] - 3 * unc[e])
                - COST_PER_S * a.duration
                - CLEAR_PENALTY * max(0.0, CLEAR_WANT - clear[e])
            )
            why = (f"separates {sep[e]:.0%} of hypothesis pairs; measures {measure[e]:.0%}; "
                   f"{progress[e]:+.3f} toward goal; clearance {clear[e]:.3f}")
            out.append(Candidate(a, float(value), float(sep[e]), float(progress[e]), float(clear[e]), why,
                                 float(room[e]), float(move[e]), bool(novel[e]), float(away[e])))
        return out

    def route(
        self,
        belief: Belief,
        pos: tuple[float, float],
        sigma: float,
        goal: tuple[float, float],
        goal_r: float,
        weights: np.ndarray | None = None,
        max_expansions: int = ROUTE_EXPANSIONS,
        ground: GroundRule = "room",
        guard: Belief | None = None,
    ) -> Route | None:
        """A* under the leading hypothesis to within `goal_r` of `goal`, every step screened
        (`ground="rule"`: every step on verified ground, as for the manual's quick-start).
        `guard`: every survivor; the first step that moves the tip, and any tap before it, must
        also pass the shield as the agent will check it (against `guard`, shortened if need be),
        or the route leads nowhere from here. A tap is no way round: a route of a mode tap and
        then a move the shield refuses would only tap again at the next plan, and again."""
        if not len(belief):
            return None
        # The route is look-ahead (every real step is screened against every survivor), so its
        # steps are screened against the few heaviest hypotheses only: the cost of the search
        # grows with the number of hypotheses predicted at every expansion.
        keep = np.argsort(-_normalise(weights, len(belief)), kind="stable")[:ROUTE_K]
        belief = Belief(belief.comp.take(keep), belief.rates[keep], belief.presses,
                        belief.armed[keep] if belief.armed is not None else None)
        lead = 0
        prims = primitives(belief.comp.programs[lead])
        if math.dist(pos, goal) <= goal_r:
            return Route([], [pos])
        geo = self._geodesic(goal, goal_r)
        first = Belief(belief.comp, belief.rates, belief.presses, belief.armed).predict(prims, pos, self.shield.config.limits)[0]
        # Unit of the heuristic and the dedupe cell: the longest step of the slowest actuator.
        # Full-speed units make a slow region look almost reached (the search then expands
        # forever), and steps shorter than the cell would never leave their parent's cell.
        # Weighted A* needs no admissible heuristic: a route, not the shortest one.
        moves = np.linalg.norm(first.end[lead] - np.asarray(pos), axis=1)
        per_channel = [max((m for a, m in zip(prims, moves, strict=True) if ch in a.active() and m > 1e-4), default=0.0)
                       for ch in belief.comp.programs[lead].act]
        slowest = min((m for m in per_channel if m > 0), default=ROUTE_CELL * 2)
        step = max(ROUTE_MIN_STEP, slowest)

        def h(p: tuple[float, float]) -> float:
            i, j = _cell(p, geo.shape[0])
            return float(geo[i, j]) / step

        presses0 = belief.presses.copy()
        runnable: dict[tuple[int, ...], list[bool]] = {}

        def passes(p: tuple[float, float], presses: np.ndarray) -> list[bool]:
            """Guard verdicts at a node reached by taps only (the tip predicted where it is)."""
            key = tuple(int(x) for x in presses)
            if key not in runnable:
                assert guard is not None
                g = Belief(guard.comp, guard.rates, guard.presses + (presses - presses0), guard.armed)
                runnable[key] = self._runnable(prims, p, sigma, g)
            return runnable[key]

        start = (pos, presses0, [], [pos], True)
        frontier: list[tuple[float, int, tuple]] = [(h(pos), 0, start)]
        cell = max(ROUTE_MIN_STEP / 4, min(ROUTE_CELL, slowest / 2))
        seen = {_key(pos, presses0, belief, lead, cell)}
        counter = 0
        for _ in range(max_expansions):
            if not frontier:
                break
            _, _, (p, presses, acts, pts, still) = heapq.heappop(frontier)
            here = Belief(belief.comp, belief.rates, presses, belief.armed)
            preds = here.predict(prims, p, self.shield.config.limits, want_path=True)
            verdicts = self.shield.screen(prims, p, sigma, here, ground=ground, preds=preds)
            ok = passes(p, presses) if guard is not None and still else None
            for i, (a, end, (good, _)) in enumerate(zip(prims, preds[0].end[lead], verdicts, strict=True)):
                q = (float(end[0]), float(end[1]))
                if ok is not None and not ok[i]:
                    continue
                if not good or not math.isfinite(h(q)):
                    continue
                pr = presses + _presses(a)
                k = _key(q, pr, belief, lead, cell)
                if k in seen:
                    continue
                seen.add(k)
                route = (q, pr, [*acts, a], [*pts, q], still and math.dist(p, q) < MOVE_MIN)
                if math.dist(q, goal) <= goal_r:
                    return Route(route[2], route[3])
                counter += 1
                heapq.heappush(frontier, (len(route[2]) + ROUTE_WEIGHT * h(q), counter, route))
        return None

    def _runnable(self, actions: list[Action], pos: tuple[float, float], sigma: float, guard: Belief) -> list[bool]:
        """Whether `Shield.decide` would let each action run here, at its own or a shorter
        duration, with one screen call for all of them."""
        variants: list[Action] = []
        owner: list[int] = []
        for i, a in enumerate(actions):
            v = a
            while True:
                variants.append(v)
                owner.append(i)
                if v.duration <= DURATION_MIN:
                    break
                v = v.with_duration(max(DURATION_MIN, round(v.duration / 2, 2)))
        ok = [False] * len(actions)
        for i, (good, _) in zip(owner, self.shield.screen(variants, pos, sigma, guard), strict=True):
            ok[i] = ok[i] or good
        return ok

    def _geodesic(self, goal: tuple[float, float], goal_r: float) -> np.ndarray:
        """Axis-aligned distance to the goal around hazards (margin) inside the workspace, on a
        grid of ROUTE_CELL cells: the route search's heuristic (inf where cut off). Cells within
        an unknown drop's reach of a hazard cost `NEAR_COST` times more, since off verified
        ground the shield only lets the tip move along or away from hazards there."""
        key = (round(goal[0], 4), round(goal[1], 4), round(goal_r, 4))
        if key in self._geo:
            return self._geo[key]
        n = int(math.ceil(1.0 / ROUTE_CELL))
        centers = (np.indices((n, n)).transpose(1, 2, 0) + 0.5) * ROUTE_CELL
        cfg = self.shield.config
        lo, hi = cfg.workspace
        clear = self.board.hazard_distance(centers.reshape(-1, 2)).reshape(n, n)
        free = (clear >= cfg.margin) & ((centers >= lo - ROUTE_CELL) & (centers <= hi + ROUTE_CELL)).all(axis=2)
        cost = np.where(clear < cfg.margin + cfg.limits.drop_max, NEAR_COST, 1.0) * ROUTE_CELL
        dist = np.full((n, n), np.inf)
        at_goal = np.linalg.norm(centers - np.asarray(goal), axis=2) <= goal_r + ROUTE_CELL / 2
        dist[at_goal] = 0.0
        frontier = [(0.0, int(i), int(j)) for i, j in zip(*np.nonzero(at_goal), strict=True)]
        heapq.heapify(frontier)
        while frontier:
            d, i, j = heapq.heappop(frontier)
            if d > dist[i, j]:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n and free[a, b] and d + cost[a, b] < dist[a, b]:
                    dist[a, b] = d + cost[a, b]
                    heapq.heappush(frontier, (dist[a, b], a, b))
        self._geo[key] = dist
        return dist

    def choose(
        self,
        belief: Belief,
        pos: tuple[float, float],
        sigma: float,
        goal: tuple[float, float],
        goal_r: float,
        weights: np.ndarray | None = None,
        lights: bool = False,
        proposed: list[Action] | None = None,
        guard: Belief | None = None,
        tried: Sequence[tuple[Action, tuple[float, float]]] = (),
        light_now: int = 0,
    ) -> Plan:
        """The next action, with the top candidates and the reason (CONTRACTS §6 `plan`).
        `guard`: the belief the shield checks with (every survivor; default `belief`)."""
        route = self.route(belief, pos, sigma, goal, goal_r, weights, guard=guard if guard is not None else belief)
        actions = list(proposed or []) + self.menu
        first = None
        if route is not None and route.actions:
            # Off verified ground the shield shortens the route's step to a probe: plan with that.
            first = self.shield.decide(route.actions[0], pos, sigma, guard if guard is not None else belief).final
            if first is not None:
                actions.insert(0, first)
        scored = self.score(actions, belief, pos, sigma, goal, weights, lights, guard, tried, light_now)
        if not scored:
            return Plan([], None, "no candidate action passes the shield", route)
        best = max(range(len(scored)), key=lambda k: scored[k].value)
        # A route whose first step the shield refuses (the route is screened against the
        # heaviest hypotheses only, the step against every survivor) leads nowhere either.
        blocked = route is None or first is None
        stuck = blocked and scored[best].value < STUCK_VALUE
        explore = False
        if stuck:
            best = max(range(len(scored)), key=lambda k: (round(scored[k].room, 3), -scored[k].action.duration))
            explore = scored[best].room < ROOM_MIN
            if explore:
                best = max(range(len(scored)), key=lambda k: (scored[k].novel and scored[k].move >= MOVE_MIN,
                                                              round(scored[k].away, 3), -scored[k].action.duration))
        step = 0 if first is not None and scored[0].action == first else None
        resolved = max(c.separation for c in scored) < SEP_MIN
        chosen = step if step is not None and resolved else best
        escape = False
        if resolved and step is not None and _looped(pos, tried):
            # A route can keep offering opposite first steps while its later step is blocked.
            # Make real forward measurements instead of repeating that route prefix. Small
            # probes count here: a move shorter than MOVE_MIN can still cross an uncertain line.
            forward = max(range(len(scored)), key=lambda k: (scored[k].progress, scored[k].novel,
                                                            -scored[k].action.duration))
            if scored[forward].progress > 0 and forward != step:
                chosen, escape = forward, True
        top = sorted(range(len(scored)), key=lambda k: -scored[k].value)[:TOP_CANDIDATES]
        if chosen not in top:
            top.append(chosen)
        cands = [scored[k] for k in top]
        if escape:
            why = f"route repeats without net motion; taking a shield-accepted forward probe ({scored[chosen].why})"
        elif not resolved:
            why = f"survivors disagree; best experiment {scored[best].why}"
            if chosen == step:
                why += " (also the route's next step)"
        elif chosen == step:
            assert route is not None
            why = f"hypotheses agree; route step 1 of {len(route.actions)} to the goal ({scored[step].why})"
        else:
            why = ("no safe route to the goal under the leading hypothesis; " if route is None
                   else "the route's first step does not pass the shield; " if first is None
                   else "") + f"best move: {scored[best].why}"
            if explore:
                why += (f"; nothing left to learn or gain here, so exploring a move "
                        f"{'not tried here' if scored[best].novel else 'tried before'} "
                        f"({scored[best].move:.3f} predicted)")
            elif stuck:
                why += f"; nothing left to learn here, so making room ({scored[best].room:+.3f} from hazards)"
        return Plan(cands, top.index(chosen), why, route, still=max(c.move for c in scored) < MOVE_MIN)


def _looped(pos: tuple[float, float], tried: Sequence[tuple[Action, tuple[float, float]]]) -> bool:
    """Recent motion returned near its start, rather than making a detour or just noisy fixes."""
    if len(tried) < 8:
        return False
    points = np.array([*(p for _, p in tried), pos])
    net = float(np.linalg.norm(points[-1] - points[0]))
    travel = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return net < NOVEL_R and travel > max(2 * NOVEL_R, 4 * net)


def _presses(a: Action) -> np.ndarray:
    """Presses an action makes per channel (one rising edge at most)."""
    return (Batch.inputs(a) >= LEVEL_ON).astype(int)


def _cell(p: tuple[float, float], n: int) -> tuple[int, int]:
    return min(n - 1, max(0, int(p[0] / ROUTE_CELL))), min(n - 1, max(0, int(p[1] / ROUTE_CELL)))


def _key(p: tuple[float, float], presses: np.ndarray, belief: Belief, lead: int, cell: float) -> tuple:
    mode = int(presses[int(belief.comp.mode_ch[lead])]) % 2
    return (round(p[0] / cell), round(p[1] / cell), mode)


def _normalise(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, float)
    return w / w.sum()


def separation(
    end: np.ndarray, sigma: np.ndarray, w: np.ndarray, mode: np.ndarray | None = None,
    indicator: np.ndarray | None = None, light_now: int = 0,
) -> np.ndarray:
    """Posterior mass of hypothesis pairs each experiment separates, (H,E) → (E,).

    A pair is separated by how far apart its predicted end points are in units of the combined
    standard deviation (smoothly: 1 - exp(-(d / SEP_SCALE)^4)), or fully when a tracked light is
    predicted differently (the light shows the mode under indicator programs; without the rule it
    stays as it is now, `light_now`, as the verifier scores it)."""
    h = len(w)
    if h < 2:
        return np.zeros(end.shape[1])
    diff = end[:, None] - end[None, :]  # (H,H,E,2)
    var = sigma[:, None] ** 2 + sigma[None, :] ** 2 + 2 * OBS_SIGMA**2
    d2 = (diff**2).sum(-1) / var
    sep = 1.0 - np.exp(-((d2 / SEP_SCALE**2) ** 2))
    if mode is not None and indicator is not None:
        light = np.where(indicator[:, None], mode, light_now)
        sep = np.maximum(sep, (light[:, None] != light[None, :]).astype(float))
    pair = w[:, None] * w[None, :]
    np.fill_diagonal(pair, 0.0)
    return np.einsum("hg,hge->e", pair, sep) / pair.sum()

