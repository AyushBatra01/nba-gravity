import numpy as np

from src.model import (
    BALL_ID,
    RIM_ID,
    GravityConfig,
    attractor_locations,
    baseline_locations,
    gravity_coefficients,
    infer_matchups,
    split_offense_defense,
)


DEFAULT_EPS = GravityConfig().eps


class ProcessedMoment:
    """A single tracking moment prepared for the gravity model.

    This class bridges between raw SportVU-style moment data and the linear
    algebra used to estimate gravity. 
    """

    def __init__(self, moment, poss_team_id, id=None):
        self.id = id
        self.poss_team_id = poss_team_id
        self.moment = moment

        # The ball and rim are special attractors in the model. They get their
        # own synthetic IDs so they can be fitted beside real player IDs.
        self.ball = self.moment.ball_location()
        self.rim = self.moment.rim_location()

        self.off_players, self.def_players = split_offense_defense(
            self.moment,
            self.poss_team_id,
        )

        # Dictionaries are convenient when calculating residuals or matching
        # fitted coefficients back to player IDs.
        self.off_loc = attractor_locations(self.off_players, self.ball, self.rim)
        self.def_loc = {
            defender.player_id: defender.xy
            for defender in self.def_players
        }

        self.matchups = infer_matchups(self.off_players, self.def_players)

        # Cache the small arrays used repeatedly during fitting. Geometry
        # optimization calls normal-equation construction many times, so avoiding
        # repeated list/dict work here makes alternating fits much cheaper.
        self.attractor_ids = [player.player_id for player in self.off_players] + [BALL_ID, RIM_ID]
        self.attractor_xy = np.array([self.off_loc[attractor_id] for attractor_id in self.attractor_ids])
        self.defender_xy = np.array([player.xy for player in self.def_players])
        self.matchup_xy = np.array([
            self.off_players[self.matchups[defender_index]].xy
            for defender_index in range(len(self.def_players))
        ])

    def assignments(self):
        """Wrapper around the matchups list."""
        return self.matchups

    def baseline_locations(self, rim_weight=0.2):
        """Return each defender's baseline location before gravity effects."""
        return baseline_locations(
            self.off_players,
            self.def_players,
            self.matchups,
            self.rim,
            rim_weight,
        )

    def actual_locations(self):
        """Return the observed defender locations for this tracking moment."""
        return self.def_loc

    def predicted_locations(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Predict defender locations from baseline plus gravity effects."""
        predicted = {}
        baselines = self.baseline_locations(rim_weight)

        for defender in self.def_players:
            defender_id = defender.player_id

            # Start from the no-gravity basketball baseline. Every attractor
            # then pulls this location by an amount proportional to its fitted
            # gravity coefficient.
            predicted_xy = baselines[defender_id].copy()

            for attractor_id, attractor_xy in self.off_loc.items():
                coefficient = gravity_coefficients(attractor_xy, baselines[defender_id], p, eps)
                predicted_xy += gravity[attractor_id] * coefficient

            predicted[defender_id] = predicted_xy

        return predicted

    def resid(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Return actual minus predicted defender locations."""
        actual = self.actual_locations()
        predicted = self.predicted_locations(gravity, rim_weight, p, eps)

        return {
            defender_id: actual[defender_id] - predicted[defender_id]
            for defender_id in actual
        }

    def loss(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Sum squared prediction error for this moment."""
        residuals = self.resid(gravity, rim_weight, p, eps)
        return sum(float(np.dot(vec, vec)) for vec in residuals.values())

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Build the readable, unstacked linear-system rows for this moment."""
        actual = self.actual_locations()
        baselines = self.baseline_locations(rim_weight)

        rows = []
        targets = []

        for defender_id in actual:
            baseline_xy = baselines[defender_id]
            targets.append(actual[defender_id] - baseline_xy)

            row = {}
            for attractor_id, attractor_xy in self.off_loc.items():
                row[attractor_id] = gravity_coefficients(attractor_xy, baseline_xy, p, eps)
            rows.append(row)

        return rows, targets, set(self.off_loc.keys())

    def local_normal_equation_terms(self, id_to_index, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Return this moment's small contribution to the global fit.

        Each moment only involves seven fitted attractors: five offensive
        players, the ball, and the rim. Returning local terms lets `Season`
        place those 7-by-7 values into the larger season matrix directly.
        """
        indices = [id_to_index[attractor_id] for attractor_id in self.attractor_ids]

        rim_xy = np.broadcast_to(self.rim, self.defender_xy.shape)
        baseline_xy = (1 - rim_weight) * self.matchup_xy + rim_weight * rim_xy
        target = self.defender_xy - baseline_xy

        # Shape: defenders x attractors x xy. 
        displacement = self.attractor_xy[None, :, :] - baseline_xy[:, None, :]
        distance = np.linalg.norm(displacement, axis=2)
        coefficients = displacement / np.power(distance[..., None] + eps, p)

        Ax = coefficients[:, :, 0]
        Ay = coefficients[:, :, 1]
        bx = target[:, 0]
        by = target[:, 1]

        # Local terms are only 7x7 for a moment (ball, rim, 5 off. players).
        # Return indices so the caller can place them into the larger season matrix.
        AtA_local = Ax.T @ Ax + Ay.T @ Ay
        Atb_local = Ax.T @ bx + Ay.T @ by
        btb = float(np.dot(bx, bx) + np.dot(by, by))

        return indices, AtA_local, Atb_local, btb
