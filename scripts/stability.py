"""Script to see consistency of gravity values across configurations"
"""

import json
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from src.event import Season
from src.model import BALL_ID, GravityConfig
from src.solver import Solver

game_dir = base_dir / "games"

GAME_PATHS = [f for f in game_dir.iterdir() if f.suffix == ".json"]


def player_name_lookup(games):
    """Build a player_id -> readable name dictionary from one game file."""
    players = []
    for game in games:
        players.extend(game["events"][0]["visitor"]["players"])
        players.extend(game["events"][0]["home"]["players"])

    return {
        player["playerid"]: f"{player['firstname']} {player['lastname']}"
        for player in players
    }


def label_for_id(player_id, id_to_name):
    """Use human labels for synthetic IDs and real player IDs."""
    if player_id == BALL_ID:
        return "Ball"
    return id_to_name.get(player_id, f"Player {player_id}")


def count_player_coverage(season):
    """Count sampled moments and events in which each player contributes."""
    moment_counts = {player_id: 0 for player_id in season.player_list}
    event_counts = {player_id: 0 for player_id in season.player_list}

    for game in season.games:
        for event in game.events:
            event_player_ids = set()
            for moment in event.moments:
                for player_id in moment.attractor_ids:
                    moment_counts[player_id] += 1
                    event_player_ids.add(player_id)

            # Count an event once per player, even when it contains many moments.
            for player_id in event_player_ids:
                event_counts[player_id] += 1

    return moment_counts, event_counts


def print_gravity_table(
    title,
    gravity_values,
    player_ids,
    id_to_name,
    loss,
    rmse,
    config,
    moment_counts,
    event_counts,
    limit=15,
):
    """Print a sorted gravity table for one fitted model."""
    gravity_table = sorted(
        zip(player_ids, gravity_values),
        key=lambda item: item[1],
        reverse=True,
    )

    print(title)
    print("-" * len(title))
    print(f"RMSE: {rmse:.3f} court units")
    print(f"Raw SSE loss: {loss:.2f}")
    print(f"rim_weight={config.rim_weight:.4f}, decay_power={config.decay_power:.4f}")
    print()
    print(f"{'Player':25s} {'Gravity':>10s} {'Moments':>10s} {'Events':>8s}")

    for player_id, gravity in gravity_table[:limit]:
        label = label_for_id(player_id, id_to_name)
        print(
            f"{label:25s} {gravity:10.4f} "
            f"{moment_counts[player_id]:10d} {event_counts[player_id]:8d}"
        )

    print()


def print_rank_comparison(
    values1, ids1, values2, ids2, id_to_name, moment_counts, event_counts, limit=15
):
    """Compare players appearing in either configuration's top rankings."""
    ranked1 = sorted(zip(ids1, values1), key=lambda item: item[1], reverse=True)
    ranked2 = sorted(zip(ids2, values2), key=lambda item: item[1], reverse=True)

    ranks1 = {player_id: rank for rank, (player_id, _) in enumerate(ranked1, start=1)}
    ranks2 = {player_id: rank for rank, (player_id, _) in enumerate(ranked2, start=1)}
    gravity1 = dict(zip(ids1, values1))
    gravity2 = dict(zip(ids2, values2))

    # Include anyone in either top list so players entering or leaving it remain visible.
    compared_ids = {player_id for player_id, _ in ranked1[:limit]}
    compared_ids.update(player_id for player_id, _ in ranked2[:limit])

    print("Top-ranking comparison")
    print("----------------------")
    print(
        f"{'Player':25s} {'Rank 1':>7s} {'Rank 2':>7s} {'Change':>8s} "
        f"{'Gravity 1':>10s} {'Gravity 2':>10s} {'Moments':>10s} {'Events':>8s}"
    )

    for player_id in sorted(compared_ids, key=lambda item: ranks1[item]):
        # A positive change means the player moved closer to rank 1 in config 2.
        rank_change = ranks1[player_id] - ranks2[player_id]
        label = label_for_id(player_id, id_to_name)
        print(
            f"{label:25s} {ranks1[player_id]:7d} {ranks2[player_id]:7d} "
            f"{rank_change:+8d} {gravity1[player_id]:10.4f} "
            f"{gravity2[player_id]:10.4f} {moment_counts[player_id]:10d} "
            f"{event_counts[player_id]:8d}"
        )


def print_most_events(
    values1, ids1, values2, ids2, id_to_name, moment_counts, event_counts, limit=11
):
    """Show gravity estimates for the players with the most event coverage."""
    gravity1 = dict(zip(ids1, values1))
    gravity2 = dict(zip(ids2, values2))
    most_active_ids = sorted(
        ids1,
        key=lambda player_id: (event_counts[player_id], moment_counts[player_id]),
        reverse=True,
    )[:limit]

    print()
    print("Players with the most events")
    print("----------------------------")
    print(
        f"{'Player':25s} {'Events':>8s} {'Moments':>10s} "
        f"{'Gravity 1':>10s} {'Gravity 2':>10s}"
    )

    for player_id in most_active_ids:
        label = label_for_id(player_id, id_to_name)
        print(
            f"{label:25s} {event_counts[player_id]:8d} "
            f"{moment_counts[player_id]:10d} {gravity1[player_id]:10.4f} "
            f"{gravity2[player_id]:10.4f}"
        )


def main():
    games = []
    seen_game_ids = set()
    for path in GAME_PATHS:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Extract the sample data before running this script."
            )
        with path.open() as f:
            game = json.load(f)

            # Do not count duplicate copies of the same game more than once.
            if game["gameid"] in seen_game_ids:
                continue
            seen_game_ids.add(game["gameid"])
            games.append(game)

    # These parameters define the current modeling assumption:
    # - defenders have a baseline position between matchup and rim
    # - gravity effects decay with distance squared
    # - alpha keeps fitted values from becoming too large/noisy
    config1 = GravityConfig(
        rim_weight=0.2,
        decay_power=2.0,
        alpha=1.0,
    )

    config2 = GravityConfig(
        rim_weight=0.3,
        decay_power=1.6,
        alpha=0.9,
    )

    season = Season(games)
    solver = Solver(season)
    id_to_name = player_name_lookup(games)
    moment_counts, event_counts = count_player_coverage(season)

    print(f"Processed games: {len(season.games)}")
    print(f"Processed events: {sum(len(game.events) for game in season.games)}")
    print()

    values1, ids1, loss1 = solver.solve_gravity(config1)
    print_gravity_table(
        "Config 1 gravity values",
        values1,
        ids1,
        id_to_name,
        loss1,
        season.rmse_from_loss(loss1),
        config1,
        moment_counts,
        event_counts,
    )

    print()

    values2, ids2, loss2 = solver.solve_gravity(config2)
    print_gravity_table(
        "Config 2 gravity values",
        values2,
        ids2,
        id_to_name,
        loss2,
        season.rmse_from_loss(loss2),
        config2,
        moment_counts,
        event_counts,
    )

    print_rank_comparison(
        values1,
        ids1,
        values2,
        ids2,
        id_to_name,
        moment_counts,
        event_counts,
    )

    print_most_events(
        values1,
        ids1,
        values2,
        ids2,
        id_to_name,
        moment_counts,
        event_counts,
    )


if __name__ == "__main__":
    main()
