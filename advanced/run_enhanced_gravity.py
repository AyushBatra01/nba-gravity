"""Minimal example using the enhanced classes while retaining your workflow."""

import json
import sys

from advanced.event_enhanced import Season


if len(sys.argv) < 2:
    raise SystemExit("Usage: python run_enhanced_gravity.py games/GAME.json [more games]")

with_games = []
for path in sys.argv[1:]:
    with open(path) as file:
        with_games.append(json.load(file))

season = Season(with_games, gap=0.25, pad=3.0)
fit = season.fit_alternating(max_iterations=15)
print(f"rim_weight={fit.rim_weight:.4f}; p={fit.p:.4f}; converged={fit.converged}")
for player_id, gravity in sorted(fit.gravity.items(), key=lambda row: row[1], reverse=True):
    print(f"{player_id}: {gravity:.5f}")
