# NBA Gravity

This project is an exploratory attempt to build a basketball "gravity" metric from NBA player-tracking data.

The core idea is that offensive players affect where defenders stand. A high-gravity offensive player should pull defenders away from their default defensive positions more strongly than a low-gravity player. This repo turns SportVU tracking data into a simple geometric model, then estimates player-level gravity values by fitting that model to observed defender locations.

## Project Idea

For each tracking moment, the model:

1. Identifies the team with possession.
2. Splits players into offense and defense.
3. Infers defensive matchups by finding the lowest-distance assignment between the five defenders and five offensive players.
4. Defines each defender's baseline location as a blend of:
   - their matched offensive player's location
   - the rim location
5. Predicts how the defender moves away from that baseline based on gravity from:
   - each offensive player
   - the ball
   - the rim
6. Fits gravity values that minimize the difference between predicted and actual defender locations.

In rough terms:

```text
actual defender location
  ~= baseline matchup/rim location
   + pull from offensive players
   + pull from ball
   + pull from rim
```

The fitted gravity values are intended to represent how much each offensive player warps defensive positioning.

## Repository Structure

```text
.
├── games-raw/
│   └── 12.25.2015.CLE.at.GSW.7z  # raw compressed tracking data sample
├── src/
│   ├── display.py                # plotting helpers for tracking moments
│   ├── event.py                  # Event, Game, and Season wrappers
│   ├── gather.py                 # filters moments and infers possession
│   ├── model.py                  # gravity config, matchups, baselines, coefficients
│   ├── process.py                # prepares one tracking moment for the model
│   ├── solver.py                 # optional geometry/gravity fitting experiments
│   ├── unzip.py                  # generic 7z extraction helper
│   └── util.py                   # Moment and Player data wrappers
├── run_gravity_workflow.py       # simple end-to-end script for the current workflow
└── requirements.txt              # Python dependencies
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want to use the environment from Jupyter:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name nba-gravity --display-name "Python (nba-gravity)"
```

Then choose `Python (nba-gravity)` as the notebook kernel.

## Data Workflow

The repo includes a compressed sample game in `games-raw/`.

To extract it into a local `games/` folder:

```python
from src.unzip import unzip

unzip("games-raw/12.25.2015.CLE.at.GSW.7z", "games")
```

That should produce a JSON game file such as:

```text
games/0021500438.json
```

The `games/` directory is ignored by git, so extracted data stays local. This is done to avoid adding large files to git.

## Basic Usage

The easiest way to see the current workflow is:

```bash
python run_gravity_workflow.py
```

That script loads the sample game, builds a `Season`, fits gravity values with fixed geometry, then runs a short experimental alternating fit that also tunes the geometry parameters.

The same workflow in notebook form looks like this:

Load a game and construct the season object:

```python
import json

from src.event import Season
from src.model import GravityConfig
from src.solver import Solver

with open("games/0021500438.json") as f:
    game = json.load(f)

season = Season([game])
solver = Solver(season)
```

Fit gravity values:

```python
config = GravityConfig(
    rim_weight=0.2,
    decay_power=2.0,
    alpha=1.0,
)

gravity_values, player_ids, loss = solver.solve_gravity(config)
gravity = dict(zip(player_ids, gravity_values))
```

The raw loss is a sum of squared errors across every defender x/y coordinate, so it can be large even for one game. For a more readable scale, convert it to per-coordinate RMSE:

```python
rmse = season.rmse_from_loss(loss)
```

To jointly tune gravity values and geometry parameters, use the experimental alternating fit:

```python
gravity_values, player_ids, loss, fitted_config = solver.fit_alternating(
    initial_config=config,
    max_iterations=3,
    geometry_max_iterations=8,
)
```

This alternates between closed-form gravity fitting and bounded optimization of `rim_weight`/`decay_power`. Geometry scoring uses the same normal-equation pieces as the linear fit, which is faster than recomputing every predicted defender location one by one.

Special IDs are included in the fitted values:

```text
-1 = ball
-2 = rim
```

To map player IDs back to names for a game:

```python
players = season.games[0].away["players"] + season.games[0].home["players"]
id_to_name = {
    player["playerid"]: f"{player['firstname']} {player['lastname']}"
    for player in players
}
```

## Visualizing a Moment

```python
from src.display import visualize_moment

game = season.games[0]
event = game.events[0]
moment = event.logs[0]

visualize_moment(moment, event.poss_team)
```

This plots offensive players, defensive players, the ball, and the inferred rim location for a single tracking moment. The function returns the matplotlib axis, so you can also pass your own `ax` when building notebook subplots.

## Model Notes

The gravity model is intentionally simple right now:

- Defensive matchups are estimated from nearest assignment, not from lineup role or play context.
- Possession is inferred from whichever team is closest to the ball for most logged moments in an event.
- Each defender's baseline location is a weighted average of their matchup and the rim.
- Rim direction is inferred from the average non-ball player location in the moment.
- Offensive players, the ball, and the rim exert distance-decayed influence on defenders.
- Gravity fitting is solved with regularized normal equations.

The main tunable parameters live in `GravityConfig`:

- `rim_weight`: how much the baseline defender position is pulled toward the rim.
- `decay_power`: distance-decay exponent for gravity effects.
- `eps`: small distance offset to avoid unstable near-zero distances.
- `alpha`: ridge regularization strength.

The event-cleaning parameters are:

- `gap`: spacing between sampled tracking moments.
- `pad`: time trimmed from event starts and ends to avoid transition noise.

## Current Status

This is an exploratory research repo, not a finished package.

What exists:

- tracking-data wrappers for players, moments, events, games, and seasons
- possession inference
- moment filtering and basic data-quality checks
- defender matchup inference
- single-pass gravity fitting via `Season.solve_gravity`
- optional alternating geometry/gravity fitting via `Solver.fit_alternating`
- visualization for individual moments

Ideas for next steps can be seem in `EXTENSIONS.md`. 
