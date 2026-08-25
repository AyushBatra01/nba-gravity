"""Fast, explicit implementation of the defensive-displacement gravity model.

This file is intentionally independent of the original modules.  It reads the
same NBA SportVU JSON format, but turns selected moments into compact NumPy
arrays once, so fitting several (rim_weight, p) combinations does not need to
re-parse every event.

Coordinates are in feet.  Player ids -1 and -2 refer to the ball and rim;
they can either be fitted like normal sources or supplied in ``fixed_gravity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr


COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
BALL_ID = -1
RIM_ID = -2


@dataclass(frozen=True)
class FitIteration:
    iteration: int
    rim_weight: float
    distance_exponent: float
    squared_error: float
    mass_change: float
    parameter_change: float


@dataclass
class AlternatingFit:
    gravity: dict[int, float]
    rim_weight: float
    distance_exponent: float
    history: list[FitIteration]
    converged: bool


@dataclass
class TrackingBatch:
    """Selected defensive observations from one or more SportVU games.

    ``offense`` and ``defense`` have shape (frames, 5, 2).  ``source_ids``
    contains the five offensive players plus ball and rim for each frame.
    """

    offense: np.ndarray
    defense: np.ndarray
    balls: np.ndarray
    matched_offense: np.ndarray
    rims: np.ndarray
    source_ids: np.ndarray
    player_ids: list[int]

    @property
    def frame_count(self) -> int:
        return len(self.offense)

    @classmethod
    def from_games(
        cls,
        games: Iterable[Mapping],
        *,
        gap: float = 0.25,
        pad: float = 3.0,
        min_event_frames: int = 5,
        max_speed: float = 50.0,
    ) -> "TrackingBatch":
        """Build a reusable batch from decoded game JSON dictionaries.

        Each event is trimmed independently by ``pad`` seconds at both ends.
        This is safer than carrying a game-clock bound across event records,
        which may include timeouts and duplicated moments.
        """
        offense, defense, balls, matched, rims, source_ids = [], [], [], [], [], []
        for game in games:
            for event in game.get("events", []):
                frames = _selected_event_frames(
                    event.get("moments", []), gap=gap, pad=pad, max_speed=max_speed
                )
                if len(frames) < min_event_frames:
                    continue
                possession = _possession_team(frames)
                if possession is None:
                    continue
                for locations in frames:
                    parsed = _split_locations(locations, possession)
                    if parsed is None:
                        continue
                    off_xy, off_ids, def_xy, _def_ids, ball_xy = parsed
                    # The offense's side identifies its attacking basket.  Using
                    # all ten players here can incorrectly select the far basket.
                    rim = np.array(
                        [COURT_LENGTH if off_xy[:, 0].mean() >= COURT_LENGTH / 2 else 0.0,
                         COURT_WIDTH / 2],
                        dtype=float,
                    )
                    rows, cols = linear_sum_assignment(
                        ((def_xy[:, None, :] - off_xy[None, :, :]) ** 2).sum(axis=2)
                    )
                    assignment = np.empty(5, dtype=np.int8)
                    assignment[rows] = cols
                    offense.append(off_xy)
                    defense.append(def_xy)
                    balls.append(ball_xy)
                    matched.append(assignment)
                    rims.append(rim)
                    source_ids.append(np.r_[off_ids, BALL_ID, RIM_ID])

        if not offense:
            raise ValueError("No valid tracking frames were found.")
        source_id_array = np.asarray(source_ids, dtype=int)
        player_ids = sorted(set(source_id_array.ravel().tolist()))
        return cls(
            offense=np.asarray(offense, dtype=float),
            defense=np.asarray(defense, dtype=float),
            balls=np.asarray(balls, dtype=float),
            matched_offense=np.asarray(matched, dtype=np.int8),
            rims=np.asarray(rims, dtype=float),
            source_ids=source_id_array,
            player_ids=player_ids,
        )

    def _geometry(self, rim_weight: float, distance_exponent: float, eps: float):
        if not 0.0 <= rim_weight <= 1.0:
            raise ValueError("rim_weight must be between 0 and 1.")
        if distance_exponent <= 0.0:
            raise ValueError("distance_exponent must be positive.")
        matched_xy = np.take_along_axis(
            self.offense, self.matched_offense[..., None], axis=1
        )
        baseline = (1.0 - rim_weight) * matched_xy + rim_weight * self.rims[:, None, :]
        sources = np.concatenate(
            (self.offense, np.zeros((self.frame_count, 2, 2), dtype=float)), axis=1
        )
        sources[:, 5, :] = self.balls
        sources[:, 6, :] = self.rims
        displacement = sources[:, None, :, :] - baseline[:, :, None, :]
        coefficient = displacement / (
            np.linalg.norm(displacement, axis=-1, keepdims=True) + eps
        ) ** distance_exponent
        target = self.defense - baseline
        return coefficient, target

    def design_matrix(
        self,
        rim_weight: float,
        distance_exponent: float,
        *,
        eps: float = 0.5,
        source_ids: Sequence[int] | None = None,
    ) -> tuple[csr_matrix, np.ndarray, list[int]]:
        """Return sparse A and b for ``A @ gravity ~= defender - baseline``."""
        coefficient, target = self._geometry(rim_weight, distance_exponent, eps)
        ids = self.player_ids if source_ids is None else sorted(source_ids)
        id_array = np.asarray(ids, dtype=int)
        source_column = np.searchsorted(id_array, self.source_ids)
        present = source_column < len(id_array)
        present[valid := present.copy()] &= (
            id_array[source_column[valid]] == self.source_ids[valid]
        )
        obs = np.arange(self.frame_count * 5).reshape(self.frame_count, 5, 1)
        rows = np.broadcast_to(2 * obs, coefficient.shape[:3])
        columns = np.broadcast_to(source_column[:, None, :], coefficient.shape[:3])
        keep = np.broadcast_to(present[:, None, :], coefficient.shape[:3])

        row_x = rows[keep]
        row_y = (rows + 1)[keep]
        col = columns[keep].astype(int)
        data_x = coefficient[..., 0][keep]
        data_y = coefficient[..., 1][keep]
        matrix = coo_matrix(
            (np.r_[data_x, data_y], (np.r_[row_x, row_y], np.r_[col, col])),
            shape=(self.frame_count * 5 * 2, len(ids)),
        ).tocsr()
        return matrix, target.reshape(-1), ids

    def solve_mass(
        self,
        rim_weight: float = 0.2,
        distance_exponent: float = 2.0,
        *,
        alpha: float = 1e-3,
        eps: float = 0.5,
        fixed_gravity: Mapping[int, float] | None = None,
    ) -> dict[int, float]:
        """Ridge-regression gravity fit, optionally holding ball/rim values fixed."""
        fixed_gravity = dict(fixed_gravity or {})
        fitted_ids = [pid for pid in self.player_ids if pid not in fixed_gravity]
        matrix, target, fitted_ids = self.design_matrix(
            rim_weight, distance_exponent, eps=eps, source_ids=fitted_ids
        )
        if fixed_gravity:
            fixed_matrix, _, fixed_ids = self.design_matrix(
                rim_weight, distance_exponent, eps=eps, source_ids=list(fixed_gravity)
            )
            fixed_values = np.array([fixed_gravity[pid] for pid in fixed_ids])
            target = target - fixed_matrix @ fixed_values
        result = lsqr(matrix, target, damp=np.sqrt(alpha), atol=1e-7, btol=1e-7)
        gravity = dict(fixed_gravity)
        gravity.update(zip(fitted_ids, result[0], strict=True))
        return gravity

    def squared_error(
        self,
        gravity: Mapping[int, float],
        rim_weight: float,
        distance_exponent: float,
        *,
        eps: float = 0.5,
    ) -> float:
        coefficient, target = self._geometry(rim_weight, distance_exponent, eps)
        # ``player_ids`` is sorted, so searchsorted turns the id-to-mass lookup
        # into vectorized NumPy indexing instead of a Python loop per frame.
        player_ids = np.asarray(self.player_ids, dtype=int)
        values_by_id = np.array([gravity.get(int(pid), 0.0) for pid in player_ids])
        source_columns = np.searchsorted(player_ids, self.source_ids)
        values = values_by_id[source_columns]
        residual = target - (coefficient * values[:, None, :, None]).sum(axis=2)
        return float(np.square(residual).sum())

    def optimize_geometry(
        self,
        gravity: Mapping[int, float],
        *,
        initial_rim_weight: float,
        initial_distance_exponent: float,
        rim_weight_bounds: tuple[float, float] = (0.0, 0.8),
        exponent_bounds: tuple[float, float] = (0.5, 4.0),
        eps: float = 0.5,
    ) -> tuple[float, float]:
        """Optimize geometry while treating ``gravity`` as fixed."""
        result = minimize(
            lambda x: self.squared_error(gravity, x[0], x[1], eps=eps),
            x0=np.array([initial_rim_weight, initial_distance_exponent]),
            method="L-BFGS-B",
            bounds=(rim_weight_bounds, exponent_bounds),
        )
        if not result.success:
            raise RuntimeError(f"Geometry optimization failed: {result.message}")
        return float(result.x[0]), float(result.x[1])

    def fit_alternating(
        self,
        *,
        initial_rim_weight: float = 0.2,
        initial_distance_exponent: float = 2.0,
        alpha: float = 1e-3,
        fixed_gravity: Mapping[int, float] | None = None,
        max_iterations: int = 20,
        tolerance: float = 1e-3,
        rim_weight_bounds: tuple[float, float] = (0.0, 0.8),
        exponent_bounds: tuple[float, float] = (0.5, 4.0),
        eps: float = 0.5,
    ) -> AlternatingFit:
        """Alternate between ridge mass fitting and bounded geometry fitting.

        Stopping requires *both* a small relative mass change and a small
        geometry change.  The final masses are always re-fit at final geometry.
        """
        rim_weight, exponent = initial_rim_weight, initial_distance_exponent
        gravity = self.solve_mass(rim_weight, exponent, alpha=alpha, eps=eps,
                                  fixed_gravity=fixed_gravity)
        history: list[FitIteration] = []
        for iteration in range(1, max_iterations + 1):
            new_weight, new_exponent = self.optimize_geometry(
                gravity,
                initial_rim_weight=rim_weight,
                initial_distance_exponent=exponent,
                rim_weight_bounds=rim_weight_bounds,
                exponent_bounds=exponent_bounds,
                eps=eps,
            )
            new_gravity = self.solve_mass(
                new_weight, new_exponent, alpha=alpha, eps=eps,
                fixed_gravity=fixed_gravity,
            )
            old_values = np.array([gravity[pid] for pid in sorted(gravity)])
            new_values = np.array([new_gravity[pid] for pid in sorted(gravity)])
            mass_change = float(np.linalg.norm(new_values - old_values) /
                                max(1.0, np.linalg.norm(old_values)))
            parameter_change = float(max(abs(new_weight - rim_weight),
                                         abs(new_exponent - exponent)))
            history.append(FitIteration(
                iteration, new_weight, new_exponent,
                self.squared_error(new_gravity, new_weight, new_exponent, eps=eps),
                mass_change, parameter_change,
            ))
            gravity, rim_weight, exponent = new_gravity, new_weight, new_exponent
            if mass_change < tolerance and parameter_change < tolerance:
                return AlternatingFit(gravity, rim_weight, exponent, history, True)
        return AlternatingFit(gravity, rim_weight, exponent, history, False)


def _selected_event_frames(moments, *, gap: float, pad: float, max_speed: float):
    """Validate, trim, and downsample one event without creating Moment objects."""
    if not moments:
        return []
    start_clock, end_clock = moments[0][2], moments[-1][2]
    previous_clock = start_clock
    selected = []
    previous_locations = None
    for raw in moments:
        if len(raw) < 6 or len(raw[5]) != 11:
            continue
        clock, locations = raw[2], raw[5]
        if start_clock - clock <= pad or clock - end_clock <= pad:
            continue
        if previous_clock - clock + 1e-9 < gap:
            continue
        if not _valid_locations(locations, previous_locations, previous_clock - clock, max_speed):
            continue
        selected.append(locations)
        previous_locations, previous_clock = locations, clock
    return selected


def _valid_locations(locations, previous, elapsed, max_speed):
    ids = [entry[1] for entry in locations]
    if len(set(ids)) != 11 or ids[0] != BALL_ID:
        return False
    player_xy = [(entry[2], entry[3]) for entry in locations[1:]]
    if len(set(player_xy)) != 10:
        return False
    if previous is None or elapsed <= 0:
        return True
    prior = {entry[1]: np.asarray(entry[2:4], dtype=float) for entry in previous}
    for entry in locations:
        if entry[1] not in prior:
            return False
        if np.linalg.norm(np.asarray(entry[2:4]) - prior[entry[1]]) > elapsed * max_speed:
            return False
    return True


def _possession_team(frames):
    closest_teams = []
    for locations in frames:
        ball = np.asarray(locations[0][2:4], dtype=float)
        players = locations[1:]
        distances = [np.linalg.norm(np.asarray(player[2:4]) - ball) for player in players]
        closest_teams.append(players[int(np.argmin(distances))][0])
    teams, counts = np.unique(closest_teams, return_counts=True)
    return int(teams[np.argmax(counts)]) if len(teams) else None


def _split_locations(locations, possession):
    ball = np.asarray(locations[0][2:4], dtype=float)
    offense = [entry for entry in locations[1:] if entry[0] == possession]
    defense = [entry for entry in locations[1:] if entry[0] != possession]
    if len(offense) != 5 or len(defense) != 5:
        return None
    return (
        np.asarray([entry[2:4] for entry in offense], dtype=float),
        np.asarray([entry[1] for entry in offense], dtype=int),
        np.asarray([entry[2:4] for entry in defense], dtype=float),
        np.asarray([entry[1] for entry in defense], dtype=int),
        ball,
    )
