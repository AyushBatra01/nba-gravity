"""Example workflow for the current NBA gravity model.

Run this file from the repo root:

    python run_gravity_workflow.py

If `games/0021500438.json` does not exist yet, extract the sample data first:

    python -c "from src.unzip import unzip; unzip('games-raw/12.25.2015.CLE.at.GSW.7z', 'games')"
"""

import json
from pathlib import Path

from src.event import Season
from src.model import BALL_ID, RIM_ID, GravityConfig
from src.solver import Solver


GAME_PATH = Path("games/0021500438.json")


def player_name_lookup(game):
    """Build a player_id -> readable name dictionary from one game file."""
    players = game["events"][0]["visitor"]["players"] + game["events"][0]["home"]["players"]

    return {
        player["playerid"]: f"{player['firstname']} {player['lastname']}"
        for player in players
    }


def label_for_id(player_id, id_to_name):
    """Use human labels for synthetic IDs and real player IDs."""
    if player_id == BALL_ID:
        return "Ball"
    if player_id == RIM_ID:
        return "Rim"
    return id_to_name.get(player_id, f"Player {player_id}")


def print_gravity_table(
    title,
    gravity_values,
    player_ids,
    id_to_name,
    loss,
    rmse,
    config,
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

    for player_id, gravity in gravity_table[:limit]:
        label = label_for_id(player_id, id_to_name)
        print(f"{label:25s} {gravity: .4f}")

    print()


def main():
    if not GAME_PATH.exists():
        raise FileNotFoundError(
            f"{GAME_PATH} does not exist. Extract the sample data before running this script."
        )

    with GAME_PATH.open() as f:
        game = json.load(f)

    # These parameters define the current modeling assumption:
    # - defenders have a baseline position between matchup and rim
    # - gravity effects decay with distance squared
    # - alpha keeps fitted values from becoming too large/noisy
    config = GravityConfig(
        rim_weight=0.2,
        decay_power=2.0,
        alpha=1.0,
    )

    season = Season([game])
    solver = Solver(season)
    id_to_name = player_name_lookup(game)

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
    )

    # This is slower than the fixed-parameter solve because it repeatedly
    # alternates between fitting gravity values and tuning the geometry.
    # Keep the iteration counts small while this model is still exploratory.
    alt_values, alt_ids, alt_loss, alt_config = solver.fit_alternating(
        initial_config=config,
        max_iterations=3,
        geometry_max_iterations=8,
        verbose=False,
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
    )


if __name__ == "__main__":
    main()
