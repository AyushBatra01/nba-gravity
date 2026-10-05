import numpy as np

from src.gather import get_event_logs, possession_team
from src.model import BALL_ID, RIM_ID, GravityConfig
from src.process import DEFAULT_EPS, ProcessedMoment
from src.util import Moment


MINIMUM_MOMENTS = 5


class Event:
    """One play/event after filtering and possession inference."""

    def __init__(self, event, gap=0.25, pad=3.0, bound_start=None, bound_end=None):
        self.raw = event
        self.min_moments = MINIMUM_MOMENTS

        # First turn raw tracking samples into cleaner Moment objects. We keep
        # idx so a notebook can trace a processed moment back to the raw event.
        self.logs, self.idx, self.num_filter_errors = get_event_logs(
            event,
            gap=gap,
            pad=pad,
            bound_start=bound_start,
            bound_end=bound_end,
        )

        self.poss_team = None
        self.moments = []

        if self.valid_event():
            self.poss_team = possession_team(self.logs, check_tie=True)
            self.process_moments()

    def valid_event(self):
        """An event needs enough clean moments to be useful for fitting."""
        return len(self.logs) >= self.min_moments

    def process_moments(self):
        """Convert filtered raw moments into gravity-model moments."""
        self.moments = [
            ProcessedMoment(moment, self.poss_team, moment_id)
            for moment, moment_id in zip(self.logs, self.idx)
        ]

    def loss(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Total squared error across all processed moments in this event."""
        return sum(
            moment.loss(gravity, rim_weight, p, eps)
            for moment in self.moments
        )

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Readable linear-system rows for all moments in this event."""
        rows = []
        targets = []
        attractor_ids = set()

        for moment in self.moments:
            moment_rows, moment_targets, moment_attractors = moment.matrix(rim_weight, p, eps)
            rows.extend(moment_rows)
            targets.extend(moment_targets)
            attractor_ids |= moment_attractors

        return rows, targets, attractor_ids


class Game:
    """A full game represented as valid processed events."""

    def __init__(self, game, gap=0.25, pad=3.0):
        self.raw = game
        self.id = game["gameid"]
        self.date = game["gamedate"]
        self.home = game["events"][0]["home"]
        self.away = game["events"][0]["visitor"]

        self.events = []
        self.errors = []

        # start prevents overlap between adjacent events
        start = None

        for event_index, raw_event in enumerate(game["events"]):
            if len(raw_event["moments"]) == 0:
                continue

            if Moment(raw_event["moments"][0]).game_clock == 720.0:
                start = None        # reset start at beginning of each period

            try:
                event = Event(raw_event, gap=gap, pad=pad, bound_start=start)
            except (IndexError, KeyError, ValueError) as error:
                self.errors.append((event_index, error))
                continue

            if event.valid_event():
                self.events.append(event)
                start = event.logs[-1].game_clock

    def loss(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Total squared error across all valid events in this game."""
        return sum(
            event.loss(gravity, rim_weight, p, eps)
            for event in self.events
        )

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Readable linear-system rows for all events in this game."""
        rows = []
        targets = []
        attractor_ids = set()

        for event in self.events:
            event_rows, event_targets, event_attractors = event.matrix(rim_weight, p, eps)
            rows.extend(event_rows)
            targets.extend(event_targets)
            attractor_ids |= event_attractors

        return rows, targets, attractor_ids


class Season:
    """Collection of games plus fitting helpers for player gravity."""

    def __init__(self, games, gap=0.25, pad=3.0):
        self.games = [Game(game, gap=gap, pad=pad) for game in games]

        # keep track of all unique offensive players across the season, plus the
        # ball and rim. This is the list of players we fit gravity for.
        offensive_player_ids = {
            player.player_id
            for game in self.games
            for event in game.events
            for moment in event.moments
            for player in moment.off_players
        }

        self.player_list = sorted(offensive_player_ids | {BALL_ID, RIM_ID})
        self.id_to_index = {
            player_id: index
            for index, player_id in enumerate(self.player_list)
        }
        self.num_moments = sum(
            len(event.moments)
            for game in self.games
            for event in game.events
        )
        # Every moment has five defenders, and each defender contributes an x
        # residual and a y residual. This lets us report RMSE in court units
        # instead of only showing a giant sum of squared errors.
        self.num_residual_coordinates = 2 * 5 * self.num_moments

    def loss(self, gravity, rim_weight=0.2, p=2, alpha=1e-3, eps=DEFAULT_EPS, with_alpha=True):
        """Total model loss, optionally including ridge regularization."""
        total = sum(
            game.loss(gravity, rim_weight, p, eps)
            for game in self.games
        )

        if with_alpha:
            gravity_values = np.array([gravity[player_id] for player_id in self.player_list])
            total += alpha * np.dot(gravity_values, gravity_values)

        return total

    def normal_equation_terms(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        """Aggregate A.T @ A, A.T @ b, and b.T @ b across the season."""
        n_players = len(self.player_list)
        AtA = np.zeros((n_players, n_players))
        Atb = np.zeros(n_players)
        btb = 0.0

        for game in self.games:
            for event in game.events:
                for moment in event.moments:
                    indices, AtA_m, Atb_m, btb_m = moment.local_normal_equation_terms(
                        self.id_to_index,
                        rim_weight=rim_weight,
                        p=p,
                        eps=eps,
                    )
                    # Insert the small moment-level system into the global season system. 
                    AtA[np.ix_(indices, indices)] += AtA_m
                    Atb[indices] += Atb_m
                    btb += btb_m

        return AtA, Atb, btb, self.player_list

    def loss_from_terms(self, gravity_values, AtA, Atb, btb, alpha=1e-3):
        """Compute squared-error loss from normal-equation pieces.

        This equals ||A g - b||^2 + alpha ||g||^2, but avoids materializing A.
        It is useful during geometry optimization, where we evaluate many
        candidate geometries while holding the current gravity values fixed.
        """
        return (
            btb
            - 2 * gravity_values @ Atb
            + gravity_values @ AtA @ gravity_values
            + alpha * np.dot(gravity_values, gravity_values)
        )

    def rmse_from_loss(self, loss):
        """Scale total loss to per-coordinate RMSE in court units."""
        return np.sqrt(loss / self.num_residual_coordinates)

    def solve_gravity(self, config=None):
        """Fit gravity coefficients with ridge-regularized least squares."""
        if config is None:
            config = GravityConfig()

        AtA, Atb, btb, player_list = self.normal_equation_terms(
            rim_weight=config.rim_weight,
            p=config.decay_power,
            eps=config.eps,
        )

        regularizer = config.alpha * np.eye(AtA.shape[0])
        gravity_values = np.linalg.solve(AtA + regularizer, Atb)

        # This is the regularized objective value at the fitted solution.
        loss = self.loss_from_terms(gravity_values, AtA, Atb, btb, config.alpha)

        return gravity_values, player_list, loss
