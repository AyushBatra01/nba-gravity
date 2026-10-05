from dataclasses import dataclass
from itertools import permutations

import numpy as np

from src.util import SCALE


BALL_ID = -1
RIM_ID = -2


@dataclass(frozen=True)
class GravityConfig:
    """Parameters that define one version of the gravity model."""

    rim_weight: float = 0.2
    decay_power: float = 2.0
    eps: float = 0.5 * SCALE
    alpha: float = 1e-3


def split_offense_defense(moment, poss_team_id):
    """Split a raw tracking moment into offensive and defensive players."""
    offensive_players = []
    defensive_players = []

    for player in moment.locations:
        # handle ball separately from players
        if player.team_id == BALL_ID:
            continue
        if player.team_id == poss_team_id:
            offensive_players.append(player)
        else:
            defensive_players.append(player)

    return offensive_players, defensive_players


def infer_matchups(offensive_players, defensive_players):
    """Assign each defender to one offensive player using minimum distance."""
    if len(offensive_players) != 5 or len(defensive_players) != 5:
        raise ValueError("Expected exactly 5 offensive and 5 defensive players.")

    offensive_xy = np.array([player.xy for player in offensive_players])
    defensive_xy = np.array([player.xy for player in defensive_players])

    best_matchups = None
    best_cost = np.inf

    # Do brute force since 5! not too large. 
    for matchups in permutations(range(5)):
        cost = 0.0
        for defender_index, offensive_index in enumerate(matchups):
            displacement = defensive_xy[defender_index] - offensive_xy[offensive_index]
            cost += np.sum(displacement**2)

        if cost < best_cost:
            best_cost = cost
            best_matchups = matchups

    return list(best_matchups)


def baseline_locations(offensive_players, defensive_players, matchups, rim_xy, rim_weight):
    """Compute each defender's no-gravity baseline location.

    Assumption is that absent any gravity effects, a defender stands somewhere
    between their matchup and the rim. The rim_weight parameter controls how
    much the defender is pulled toward the rim.
    """
    baselines = {}

    for defender_index, defender in enumerate(defensive_players):
        matched_offender = offensive_players[matchups[defender_index]]

        # rim_weight=0 means "stand on the matchup"; rim_weight=1 means
        # "stand at the rim." Values between those extremes represent normal
        # help-defense positioning.
        baselines[defender.player_id] = (
            (1 - rim_weight) * matched_offender.xy
            + rim_weight * rim_xy
        )

    return baselines


def attractor_locations(offensive_players, ball_xy, rim_xy):
    """Return the objects that can pull defenders away from baseline."""
    locations = {player.player_id: player.xy for player in offensive_players}
    locations[BALL_ID] = ball_xy
    locations[RIM_ID] = rim_xy
    return locations


def gravity_coefficients(attractor_xy, baseline_xy, decay_power, eps):
    """Convert one attractor location into a linear-system coefficient.

    Model: defendder displacement ~= gravity_value * coefficient

    The coefficient points from the defender baseline toward the attractor, and
    its size shrinks as the attractor gets farther away.
    """
    displacement = attractor_xy - baseline_xy
    distance = np.linalg.norm(displacement)
    return displacement / np.power(distance + eps, decay_power)
