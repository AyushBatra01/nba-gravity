"""A drop-in-friendly, faster version of ``process.ProcessedMoment``.

The public attributes and methods deliberately mirror the original class.  The
only required caller change is importing ``ProcessedMoment`` from this module
instead of ``process`` (or, more conveniently, importing ``Season`` from
``event_enhanced``).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from util import COURT_LENGTH, COURT_WIDTH, Moment, SCALE


DEFAULT_EPS = 0.5 * SCALE
BALL_ID = -1
RIM_ID = -2


class ProcessedMoment:
    """One frame, with defensive matchups and gravity-model design terms."""

    def __init__(self, moment: Moment, poss_team_id: int, id=None):
        self.id = id
        self.poss_team_id = poss_team_id
        self.moment = moment
        self.off_players = [p for p in moment.locations if p.team_id == poss_team_id]
        self.def_players = [
            p for p in moment.locations if p.team_id not in (poss_team_id, BALL_ID)
        ]
        if len(self.off_players) != 5 or len(self.def_players) != 5:
            raise ValueError("A processed moment must contain five offensive and defensive players.")
        self.ball = moment.ball_location()
        self.rim = self._offensive_rim_location()
        self.assignments()
        # self.distance_matrix()  # Preserved for callers that use this attribute.

    def _offensive_rim_location(self) -> np.ndarray:
        """Select the basket being attacked, based on offensive players only."""
        rim_x = (SCALE * COURT_LENGTH if np.mean([p.xy[0] for p in self.off_players])
                 >= SCALE * COURT_LENGTH / 2 else 0)
        return np.array([rim_x, SCALE * COURT_WIDTH / 2])

    def assignments(self):
        """Minimum-distance one-to-one matchup assignment.

        This replaces enumeration of all 120 permutations while retaining the
        exact squared-distance objective used in the original implementation.
        """
        def_pos = np.array([player.xy for player in self.def_players])
        off_pos = np.array([player.xy for player in self.off_players])
        cost = ((def_pos[:, None, :] - off_pos[None, :, :]) ** 2).sum(axis=2)
        defenders, offensive_players = linear_sum_assignment(cost)
        matchups = np.empty(5, dtype=int)
        matchups[defenders] = offensive_players
        self.matchups = matchups.tolist()
        return self.matchups

    # def distance_matrix(self):
    #     """Retain the original distance/displacement dictionaries for compatibility."""
    #     dist, disp = {}, {}
    #     for defender in self.def_players:
    #         def_id = defender.player_id
    #         sources = {player.player_id: player.xy for player in self.off_players}
    #         sources[BALL_ID] = self.ball
    #         sources[RIM_ID] = self.rim
    #         disp[def_id] = {source_id: xy - defender.xy for source_id, xy in sources.items()}
    #         dist[def_id] = {source_id: np.linalg.norm(vector) for source_id, vector in disp[def_id].items()}
    #     self.dist_matrix, self.disp_matrix = dist, disp
    #     return dist, disp

    def baseline_locations(self, rim_weight=0.2):
        if not 0.0 <= rim_weight <= 1.0:
            raise ValueError("rim_weight must be between 0 and 1.")
        return {
            defender.player_id: (1 - rim_weight) * self.off_players[self.matchups[i]].xy
            + rim_weight * self.rim
            for i, defender in enumerate(self.def_players)
        }

    def design_terms(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Return the vectorized equivalent of this moment's ``matrix`` output.

        Returns ``(source_ids, coefficient, target)`` with shapes ``(7,)``,
        ``(5, 7, 2)``, and ``(5, 2)``.  Keeping this representation lets the
        season build a sparse matrix without Python dicts or a dense design.
        """
        # if p <= 0:
        #     raise ValueError("p must be positive.")
        matched = np.array([self.off_players[index].xy for index in self.matchups])
        baseline = (1 - rim_weight) * matched + rim_weight * self.rim
        sources = np.vstack(([player.xy for player in self.off_players], self.ball, self.rim))
        source_ids = np.array([player.player_id for player in self.off_players] + [BALL_ID, RIM_ID])
        displacement = sources[None, :, :] - baseline[:, None, :]
        coefficient = displacement / (np.linalg.norm(displacement, axis=2, keepdims=True) + eps) ** p
        actual = np.array([player.xy for player in self.def_players])
        return source_ids, coefficient, actual - baseline

    def predicted_locations(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        source_ids, coefficient, _ = self.design_terms(rim_weight, p, eps)
        masses = np.array([gravity.get(int(source_id), 0.0) for source_id in source_ids])
        predicted = np.array([self.baseline_locations(rim_weight)[p.player_id] for p in self.def_players])
        predicted += (coefficient * masses[None, :, None]).sum(axis=1)
        return {player.player_id: predicted[i] for i, player in enumerate(self.def_players)}

    def actual_locations(self):
        return {player.player_id: player.xy for player in self.def_players}

    def resid(self, gravity, rim_weight=0.2, p=2):
        predicted = self.predicted_locations(gravity, rim_weight, p)
        return {player.player_id: player.xy - predicted[player.player_id]
                for player in self.def_players}

    def loss(self, gravity, rim_weight=0.2, p=2):
        residual = self.resid(gravity, rim_weight, p)
        return sum(np.dot(vector, vector) for vector in residual.values())

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Same return format as the original: list of coefficient dicts, b, ids."""
        source_ids, coefficient, target = self.design_terms(rim_weight, p, eps)
        coefficients = [
            {int(source_id): coefficient[defender, source] for source, source_id in enumerate(source_ids)}
            for defender in range(5)
        ]
        return coefficients, list(target), set(source_ids.tolist())
