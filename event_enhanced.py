"""Enhanced versions of Event, Game, and Season that preserve the original API.

Use this module alongside the existing code; it does not modify ``event.py``.
The main improvement is ``Season.sparse_matrix``/``solve_mass``, which avoid
building the large dense matrix used by the original ``Season.matrix``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr

from gather import get_event_logs, possession_team
from process_enhanced import DEFAULT_EPS, ProcessedMoment
from util import Moment


MINIMUM_MOMENTS = 5


@dataclass(frozen=True)
class FitIteration:
    iteration: int
    rim_weight: float
    p: float
    squared_error: float
    mass_change: float
    parameter_change: float


@dataclass
class AlternatingFit:
    gravity: dict[int, float]
    rim_weight: float
    p: float
    history: list[FitIteration]
    converged: bool


class Event:
    """Equivalent to the original Event, but uses ProcessedMoment enhancement."""

    def __init__(self, event, gap=0.25, pad=3.0, bound_start=None, bound_end=None):
        self.raw = event
        self.logs, self.idx, self.n_errors = get_event_logs(
            event, gap, pad, bound_start, bound_end
        )
        self.min_moments = MINIMUM_MOMENTS
        self.poss_team = possession_team(self.logs, check_tie=True) if self.valid_event() else None
        self.moments = []

    def valid_event(self):
        return len(self.logs) >= self.min_moments

    def process_moments(self):
        self.moments = [
            ProcessedMoment(moment, self.poss_team, moment_id)
            for moment, moment_id in zip(self.logs, self.idx)
        ]

    def loss(self, gravity, rim_weight=0.2, p=2):
        return sum(moment.loss(gravity, rim_weight, p) for moment in self.moments)

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        A, b, player_set = [], [], set()
        for moment in self.moments:
            moment_A, moment_b, players = moment.matrix(rim_weight, p, eps)
            A.extend(moment_A)
            b.extend(moment_b)
            player_set |= players
        return A, b, player_set


class Game:
    """Equivalent to the original Game, retaining its game/event structure."""

    def __init__(self, game, gap=0.25, pad=3.0):
        self.id = game["gameid"]
        self.date = game["gamedate"]
        self.home = game["events"][0]["home"]
        self.away = game["events"][0]["visitor"]
        self.events, self.errors = [], []
        start = None
        for event_index, raw_event in enumerate(game["events"]):
            if not raw_event["moments"]:
                continue
            if Moment(raw_event["moments"][0]).game_clock == 720.0:
                start = None
            try:
                event = Event(raw_event, gap, pad, start)
                if event.valid_event():
                    event.process_moments()
                    self.events.append(event)
                    start = event.logs[-1].game_clock
            except (IndexError, KeyError, TypeError, ValueError) as error:
                # Preserve the original behavior of skipping damaged events, but
                # keep the reason available for inspection rather than swallowing it.
                self.errors.append((event_index, str(error)))

    def loss(self, gravity, rim_weight=0.2, p=2):
        return sum(event.loss(gravity, rim_weight, p) for event in self.events)

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        A, b, player_set = [], [], set()
        for event in self.events:
            event_A, event_b, players = event.matrix(rim_weight, p, eps)
            A.extend(event_A)
            b.extend(event_b)
            player_set |= players
        return A, b, player_set


class Season:
    """A Season with the familiar API plus sparse fitting and alternating fit."""

    def __init__(self, games, gap=0.25, pad=3.0):
        self.games = [Game(game, gap, pad) for game in games]
        self.A = self.b = self.player_list = None
        self._geometry_cache = None

    def _moments(self):
        for game in self.games:
            for event in game.events:
                yield from event.moments

    def loss(self, gravity, rim_weight=0.2, p=2):
        return sum(game.loss(gravity, rim_weight, p) for game in self.games)

    def _cached_geometry_arrays(self):
        """Cache moment data once for fast repeated geometry optimization.

        This is deliberately a Season-level implementation detail: the public
        Event and ProcessedMoment APIs stay object-oriented and easy to inspect.
        """
        if self._geometry_cache is None:
            offense, defense, balls, rims, matchups, source_ids = [], [], [], [], [], []
            for moment in self._moments():
                offense.append([player.xy for player in moment.off_players])
                defense.append([player.xy for player in moment.def_players])
                balls.append(moment.ball)
                rims.append(moment.rim)
                matchups.append(moment.matchups)
                source_ids.append([player.player_id for player in moment.off_players] + [-1, -2])
            sources = np.asarray(source_ids, dtype=int)
            self._geometry_cache = {
                "offense": np.asarray(offense, dtype=float),
                "defense": np.asarray(defense, dtype=float),
                "balls": np.asarray(balls, dtype=float),
                "rims": np.asarray(rims, dtype=float),
                "matchups": np.asarray(matchups, dtype=int),
                "source_ids": sources,
                "player_ids": np.asarray(sorted(set(sources.ravel().tolist())), dtype=int),
            }
        return self._geometry_cache

    def squared_error(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Vectorized equivalent of ``loss`` for repeated optimizer evaluations."""
        if not 0 <= rim_weight <= 1 or p <= 0:
            return float("inf")
        cache = self._cached_geometry_arrays()
        offense, defense = cache["offense"], cache["defense"]
        matched = np.take_along_axis(offense, cache["matchups"][..., None], axis=1)
        baseline = (1 - rim_weight) * matched + rim_weight * cache["rims"][:, None, :]
        sources = np.concatenate(
            (offense, cache["balls"][:, None, :], cache["rims"][:, None, :]), axis=1
        )
        displacement = sources[:, None, :, :] - baseline[:, :, None, :]
        coefficients = displacement / (np.linalg.norm(displacement, axis=-1, keepdims=True) + eps) ** p
        player_ids = cache["player_ids"]
        values_by_id = np.array([gravity.get(int(player_id), 0.0) for player_id in player_ids])
        source_columns = np.searchsorted(player_ids, cache["source_ids"])
        masses = values_by_id[source_columns]
        residual = defense - baseline - (coefficients * masses[:, None, :, None]).sum(axis=2)
        return float(np.square(residual).sum())

    def _all_player_ids(self):
        return sorted({int(source_id) for moment in self._moments()
                       for source_id in moment.design_terms()[0]})

    def sparse_matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS, source_ids=None):
        """Build sparse A and b; use this for fitting large groups of games."""
        player_list = self._all_player_ids() if source_ids is None else sorted(source_ids)
        player_array = np.asarray(player_list, dtype=int)
        rows, columns, values, targets = [], [], [], []
        row_offset = 0
        for moment in self._moments():
            moment_ids, coefficient, target = moment.design_terms(rim_weight, p, eps)
            source_columns = np.searchsorted(player_array, moment_ids)
            present = source_columns < len(player_array)
            valid = present.copy()
            present[valid] &= player_array[source_columns[valid]] == moment_ids[valid]
            # Five defenders x seven sources, with separate x and y rows.
            base_rows = row_offset + 2 * np.arange(5)[:, None]
            keep = np.broadcast_to(present[None, :], (5, 7))
            cols = np.broadcast_to(source_columns[None, :], (5, 7))[keep]
            x_rows = np.broadcast_to(base_rows, (5, 7))[keep]
            y_rows = np.broadcast_to(base_rows + 1, (5, 7))[keep]
            rows.extend(np.r_[x_rows, y_rows])
            columns.extend(np.r_[cols, cols])
            values.extend(np.r_[coefficient[:, :, 0][keep], coefficient[:, :, 1][keep]])
            targets.extend(target.reshape(-1))
            row_offset += 10
        matrix = coo_matrix((values, (rows, columns)), shape=(row_offset, len(player_list))).tocsr()
        return matrix, np.asarray(targets), player_list

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Legacy dense output. Prefer ``sparse_matrix`` for real fitting."""
        matrix, target, player_list = self.sparse_matrix(rim_weight, p, eps)
        self.A, self.b, self.player_list = matrix.toarray(), target, player_list
        return self.A, self.b, player_list

    def solve_mass(self, rim_weight=0.2, p=2, alpha=1e-3, eps=DEFAULT_EPS,
                   fixed_gravity: Mapping[int, float] | None = None):
        """Fit masses with sparse ridge regression; returns ``(masses, player_list)``.

        The tuple return is intentionally the same shape as your original
        method.  Values in ``fixed_gravity`` are excluded from the regression
        and copied unchanged into the returned mass vector.
        """
        fixed_gravity = dict(fixed_gravity or {})
        fitted_ids = [pid for pid in self._all_player_ids() if pid not in fixed_gravity]
        matrix, target, fitted_ids = self.sparse_matrix(rim_weight, p, eps, fitted_ids)
        if fixed_gravity:
            fixed_matrix, _, fixed_ids = self.sparse_matrix(
                rim_weight, p, eps, list(fixed_gravity)
            )
            target = target - fixed_matrix @ np.array([fixed_gravity[pid] for pid in fixed_ids])
        masses = lsqr(matrix, target, damp=np.sqrt(alpha), atol=1e-7, btol=1e-7)[0]
        gravity = dict(fixed_gravity)
        gravity.update(zip(fitted_ids, masses, strict=True))
        player_list = sorted(gravity)
        return np.array([gravity[pid] for pid in player_list]), player_list

    def gravity_dict(self, rim_weight=0.2, p=2, alpha=1e-3, eps=DEFAULT_EPS,
                     fixed_gravity=None):
        """Convenience form of ``solve_mass`` for the new optimization methods."""
        masses, player_list = self.solve_mass(rim_weight, p, alpha, eps, fixed_gravity)
        return dict(zip(player_list, masses, strict=True))

    def optimize_geometry(self, gravity, initial_rim_weight=0.2, initial_p=2,
                          rim_weight_bounds=(0.0, 0.8), p_bounds=(0.5, 4.0)):
        """Optimize ``rim_weight`` and ``p`` while holding player masses fixed."""
        result = minimize(
            lambda x: self.squared_error(gravity, x[0], x[1]),
            x0=np.array([initial_rim_weight, initial_p]),
            method="L-BFGS-B",
            bounds=(rim_weight_bounds, p_bounds),
        )
        if not result.success:
            raise RuntimeError(f"Geometry optimization failed: {result.message}")
        return float(result.x[0]), float(result.x[1])

    def fit_alternating(self, initial_rim_weight=0.2, initial_p=2, alpha=1e-3,
                        fixed_gravity=None, max_iterations=20, tolerance=1e-3,
                        rim_weight_bounds=(0.0, 0.8), p_bounds=(0.5, 4.0)):
        """Alternate mass and geometry fitting until both quantities stabilize."""
        rim_weight, p = initial_rim_weight, initial_p
        gravity = self.gravity_dict(rim_weight, p, alpha, fixed_gravity=fixed_gravity)
        history = []
        for iteration in range(1, max_iterations + 1):
            new_weight, new_p = self.optimize_geometry(
                gravity, rim_weight, p, rim_weight_bounds, p_bounds
            )
            new_gravity = self.gravity_dict(new_weight, new_p, alpha,
                                            fixed_gravity=fixed_gravity)
            ids = sorted(gravity)
            old_values = np.array([gravity[player_id] for player_id in ids])
            new_values = np.array([new_gravity[player_id] for player_id in ids])
            mass_change = np.linalg.norm(new_values - old_values) / max(1.0, np.linalg.norm(old_values))
            parameter_change = max(abs(new_weight - rim_weight), abs(new_p - p))
            history.append(FitIteration(iteration, new_weight, new_p,
                                        self.squared_error(new_gravity, new_weight, new_p),
                                        float(mass_change), float(parameter_change)))
            gravity, rim_weight, p = new_gravity, new_weight, new_p
            if mass_change < tolerance and parameter_change < tolerance:
                return AlternatingFit(gravity, rim_weight, p, history, True)
        return AlternatingFit(gravity, rim_weight, p, history, False)
