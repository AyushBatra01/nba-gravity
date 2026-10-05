import numpy as np

from src.util import BALL_TEAM_ID, Moment, SCALE


def get_logs(game, gap=0.25, pad=3.0):
    """Collect clean sampled moments across every event in a game.

    Most modeling code goes through `Game`/`Season`, but this helper is useful
    in notebooks when you just want a flat list of moments.
    """
    logs = []
    indices = []
    errors = 0
    start = None

    for event_index, event in enumerate(game["events"]):
        if len(event["moments"]) == 0:
            continue

        # reset bounds at start of each period
        if Moment(event["moments"][0]).game_clock == 720.0:
            start = None

        event_logs, event_indices, event_errors = get_event_logs(
            event,
            gap=gap,
            pad=pad,
            bound_start=start,
        )

        logs.extend(event_logs)
        indices.extend((event_index, moment_index) for moment_index in event_indices)
        errors += event_errors

        if event_logs:
            start = event_logs[-1].game_clock

    return logs, indices, errors


def get_event_logs(event, gap=0.25, pad=3.0, bound_start=None, bound_end=None):
    """Sample clean tracking moments from one event.

    Parameters
    ----------
    event : dict
        Raw event dictionary containing a `moments` list.
    gap : float
        Minimum seconds between sampled moments.
    pad : float
        Seconds trimmed from event start/end to avoid transition noise.
    bound_start : None or float
        Optional high game-clock bound. Used to avoid overlap with the previous
        processed event in the same period.
    bound_end : None or float
        Optional low game-clock bound.
    """
    raw_moments = event.get("moments", [])
    if len(raw_moments) < 2:
        return [], [], 0

    moments = [Moment(moment) for moment in raw_moments]

    # Game clock counts down, so begin is larger than end within a normal event.
    begin = moments[0].game_clock
    end = moments[-1].game_clock

    if bound_start is not None:
        begin = min(begin, bound_start)
    if bound_end is not None:
        end = max(end, bound_end)

    logs = []
    event_indices = []
    last_sampled_clock = begin
    num_errors = 0

    for index in range(len(moments) - 1):
        moment = moments[index]
        next_moment = moments[index + 1]
        clock = moment.game_clock
        next_clock = next_moment.game_clock

        # Ignore duplicate timestamps. 
        if clock == next_clock:
            continue

        enough_start_padding = begin - clock > pad
        enough_end_padding = clock - end > pad
        enough_gap = last_sampled_clock - clock >= gap

        if enough_start_padding and enough_end_padding and enough_gap:
            if not all_moment_checks(moment, next_moment):
                num_errors += 1
                continue

            logs.append(moment)
            event_indices.append(index)
            last_sampled_clock = clock

    return logs, event_indices, num_errors


def possession_team(event_logs, check_tie=True):
    """Infer possession from which team is closest to the ball most often."""
    counts = {}
    total_closest_dist = {}

    for moment in event_logs:
        ball_xy = moment.ball_location()
        players = [player for player in moment.locations if player.team_id != BALL_TEAM_ID]

        closest_player = min(
            players,
            key=lambda player: np.linalg.norm(ball_xy - player.xy),
        )
        closest_dist = np.linalg.norm(ball_xy - closest_player.xy)
        team_id = closest_player.team_id

        counts[team_id] = counts.get(team_id, 0) + 1
        total_closest_dist[team_id] = total_closest_dist.get(team_id, 0.0) + closest_dist

    if not counts:
        raise ValueError("Cannot infer possession from an empty event.")

    top_count = max(counts.values())
    candidate_teams = [
        team_id
        for team_id, count in counts.items()
        if count == top_count
    ]

    if len(candidate_teams) == 1 or not check_tie:
        return candidate_teams[0]

    # Tie-breaker: if each team is closest equally often, pick the team whose
    # closest-player distances were smaller in total.
    return min(candidate_teams, key=lambda team_id: total_closest_dist[team_id])


def check_no_overlap(moment):
    """Reject moments with two non-ball players at exactly the same location."""
    seen_locations = set()

    for player in moment.locations:
        if player.team_id == BALL_TEAM_ID:
            continue

        location = tuple(player.xy)
        if location in seen_locations:
            return False
        seen_locations.add(location)

    return True


def check_distance(moment1, moment2, max_dist_per_sec=SCALE * 50):
    """Reject moments where a player moves an unrealistic distance."""
    clock_gap = abs(moment1.game_clock - moment2.game_clock)
    if clock_gap == 0:
        return False

    previous_locations = {
        player.player_id: player.xy
        for player in moment1.locations
    }

    for player in moment2.locations:
        if player.player_id not in previous_locations:
            return False

        distance = np.linalg.norm(previous_locations[player.player_id] - player.xy)
        if distance > clock_gap * max_dist_per_sec:
            return False

    return True


def check_players(moment1, moment2):
    """Require the same ball plus ten player IDs in adjacent moments."""
    player_ids = {player.player_id for player in moment1.locations}

    if len(player_ids) != 11:
        return False

    return all(player.player_id in player_ids for player in moment2.locations)


def all_moment_checks(moment1, moment2):
    """Run all lightweight data-quality checks used before modeling."""
    return (
        check_no_overlap(moment1)
        and check_players(moment1, moment2)
        and check_distance(moment1, moment2)
    )
