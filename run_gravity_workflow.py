"""Example workflow for the current NBA gravity model.

Run this file from the repo root:

    python run_gravity_workflow.py

If `games/0021500438.json` does not exist yet, extract the sample data first:

    python -c "from src.unzip import unzip; unzip('games-raw/12.25.2015.CLE.at.GSW.7z', 'games')"
"""

import json
from collections import Counter
from pathlib import Path

from src.event import Season
from src.model import BALL_ID, GravityConfig
from src.solver import Solver

base_dir = Path.cwd()
game_dir = base_dir / "games"

GAME_PATHS = [f for f in game_dir.iterdir() if f.suffix == ".json"]
CORE_TEAM_IDS = {1610612744, 1610612760, 1610612739, 1610612746}  # GSW, OKC, CLE, LAC


def player_name_lookup(games):
    """Build readable names for players on the four core teams."""
    players = []
    for game in games:
        for side in ("visitor", "home"):
            team = game["events"][0][side]
            if team["teamid"] in CORE_TEAM_IDS:
                players.extend(team["players"])

    return {
        player["playerid"]: f"{player['firstname']} {player['lastname']}"
        for player in players
    }


def label_for_id(player_id, id_to_name):
    """Use human labels for synthetic IDs and real player IDs."""
    if player_id == BALL_ID:
        return "Ball"
    return id_to_name.get(player_id, f"Player {player_id}")


def player_sample_counts(season):
    """Count modeled offensive moments and distinct used events per player."""
    moments, events = Counter(), Counter()
    for game in season.games:
        for event in game.events:
            event_players = set()
            for moment in event.moments:
                player_ids = {player.player_id for player in moment.off_players}
                moments.update(player_ids)
                event_players.update(player_ids)
            events.update(event_players)
    return moments, events


def print_gravity_table(
    title,
    gravity_values,
    player_ids,
    id_to_name,
    loss,
    rmse,
    config,
    sample_counts,
    limit=None,
):
    """Print core-team players, sorted by gravity; fit metrics cover all teams."""
    gravity_table = sorted(
        ((pid, value) for pid, value in zip(player_ids, gravity_values) if pid in id_to_name),
        key=lambda item: item[1],
        reverse=True,
    )

    print(title)
    print("-" * len(title))
    print(f"RMSE: {rmse:.3f} court units")
    print(f"Raw SSE loss: {loss:.2f}")
    print(f"rim_weight={config.rim_weight:.4f}, decay_power={config.decay_power:.4f}")
    print()

    moment_counts, event_counts = sample_counts
    print(f"{'Player':25s} {'Gravity':>10s} {'Off. events':>12s} {'Off. moments':>13s}")
    for player_id, gravity in gravity_table[:limit]:
        label = label_for_id(player_id, id_to_name)
        print(f"{label:25s} {gravity:10.4f} {event_counts[player_id]:12d} {moment_counts[player_id]:13d}")

    print()


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
    config = GravityConfig(
        rim_weight=0.2,
        decay_power=2.0,
        alpha=100.0,
    )

    season = Season(games)
    solver = Solver(season)
    id_to_name = player_name_lookup(games)
    sample_counts = player_sample_counts(season)

    print(f"Processed games: {len(season.games)}")
    print(f"Processed events: {sum(len(game.events) for game in season.games)}")
    print()

    fixed_values, fixed_ids, fixed_loss = solver.solve_gravity(config)
    print_gravity_table(
        "Fixed-parameter gravity values",
        fixed_values,
        fixed_ids,
        id_to_name,
        fixed_loss,
        season.rmse_from_loss(fixed_loss),
        config,
        sample_counts,
    )

    # This is slower than the fixed-parameter solve because it repeatedly
    # alternates between fitting gravity values and tuning the geometry.
    # Keep the iteration counts small while this model is still exploratory.
    alt_values, alt_ids, alt_loss, alt_config = solver.fit_alternating(
        initial_config=config,
        max_iterations=3,
        geometry_max_iterations=5,
        verbose=True,
    )
    print()
    print_gravity_table(
        "Alternating-fit gravity values",
        alt_values,
        alt_ids,
        id_to_name,
        alt_loss,
        season.rmse_from_loss(alt_loss),
        alt_config,
        sample_counts,
    )


if __name__ == "__main__":
    main()
