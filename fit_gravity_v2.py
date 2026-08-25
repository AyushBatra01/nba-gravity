"""Example command-line entry point for gravity_model_v2.py.

Usage:
    python fit_gravity_v2.py games/0021500438.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gravity_model_v2 import TrackingBatch


def main(paths: list[str]) -> None:
    if not paths:
        raise SystemExit("Usage: python fit_gravity_v2.py GAME.json [GAME2.json ...]")
    games = [json.loads(Path(path).read_text()) for path in paths]
    batch = TrackingBatch.from_games(games, gap=0.25, pad=3.0)
    # If you have a reason to hold ball/rim effects fixed, e.g.
    # fixed = {-1: 0.20, -2: 0.10}; otherwise leave them estimated.
    fit = batch.fit_alternating(max_iterations=15, tolerance=1e-3)
    print(f"Frames: {batch.frame_count}")
    print(f"rim_weight={fit.rim_weight:.5f}, p={fit.distance_exponent:.5f}")
    print(f"converged={fit.converged}, iterations={len(fit.history)}")
    print("\nGravity estimates (player id: mass):")
    for player_id, mass in sorted(fit.gravity.items(), key=lambda item: item[1], reverse=True):
        print(f"{player_id}: {mass:.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
