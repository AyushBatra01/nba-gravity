# NBA Gravity Extensions

This file lists possible extensions to the current gravity model that may be implemented in future versions. 

## Version 2

### Multi-Game Support

Run the current workflow on a larger set of games instead of only the single sample game.

The code is already structured around `Season([game1, game2, ...])`, so this is mostly about making the data workflow practical: loading multiple extracted JSON files, tracking runtime, and checking that player IDs line up cleanly across games.

This is the natural first step for Version 2 because one-game gravity values will be noisy and lineup/context dependent.

Link to other 7z files: https://github.com/sealneaward/nba-movement-data/tree/master

### Sanity Check Outputs

Create simple checks that the fitted gravity values resemble basketball reality.

Examples could include checking whether known high-usage offensive threats rank reasonably high, whether the ball's fitted gravity is sensible, whether low-minute players look overly extreme, and whether values are stable across nearby parameter choices.

This does not prove the model is correct, but it helps catch obvious failures before adding more complicated extensions.

### Efficiency Improvements

Make the alternating fit practical enough to use as part of the normal workflow.

Right now the fixed-parameter gravity fit is reasonably fast, but the alternating procedure is slower because it repeatedly refits gravity values and re-optimizes geometry parameters. Version 2 should focus on turning this from an experimental option into something that can run comfortably on multiple games.

A practical first version could optimize geometry on a random subset of moments, then refit gravity on the full dataset once the parameters are chosen. Another option is to cache more geometry-independent arrays so each candidate parameter setting can be scored with less repeated work.

The goal is making the alternating fit fast enough that it can be used while experimenting, comparing settings, and scaling to multiple games.

### Testing

Add tests for the parts of the pipeline that are easiest to accidentally break.

Good first targets are moment filtering, possession inference, matchup assignment, baseline construction, normal-equation construction, and the relationship between raw loss and RMSE. Small synthetic tracking moments would be enough for most of these tests.

This would make the project safer to extend, especially once future versions start changing model assumptions.

### Sample-Size-Aware Gravity

Shrink low-sample players more aggressively toward zero.

One way to do this is to make the ridge penalty depend on playing time, possessions, or number of modeled moments. Players with fewer observations would get a larger penalty, while high-minute players would be estimated more freely.

This is the most pressing next extension since it will address noisy one game estimates. 

## Version 3 and Beyond

### On-Ball vs Off-Ball Gravity

Estimate separate gravity values for when a player has the ball and when they are off the ball.

This would let the model distinguish players who bend the defense as ball handlers from players who create spacing without touching the ball. A simple version could add two coefficients per player: one active when the player is closest to the ball, and one active otherwise.

### Location-Dependent Gravity

Allow gravity to depend on where the offensive player is on the court.

For example, a shooter in the corner, above the break, or near the rim may pull defenders differently. A simple version could use court zones. A more flexible version could use smooth spatial basis functions.

This is more a more difficult extension, so it is not a priority. 

### Momentum and Offensive Motion

Include player velocity, especially movement toward the rim, as part of gravity.

This could help capture help-defense reactions. A player standing still at the three-point line and a player driving downhill may have the same location but very different defensive effects.

### Smoother Matchup Modeling

Replace hard defender-offender assignments with a smoother matchup model.

The current nearest-assignment approach creates discontinuities when matchups switch. A smoother approach could assign each defender partial responsibility for multiple offensive players, with weights based on distance, role, or defensive context.

This is mathematically cleaner and may make geometry optimization behave better.

### Defensive Principles

Add basketball context to the baseline defender position.

The current baseline is only a blend of matchup and rim. Real defenders also care about strong side vs weak side, help position, whether they are behind the play, ball pressure, rotations, and rim protection.

This is likely important if the model is meant to become interpretable rather than just fit tracking positions. This is also more difficult to determine from data alone, so it is a distant next step. 

### Defender Reactivity

Estimate a defensive trait for how strongly each defender responds to offensive gravity.

This is the "defensive mass" idea. The physics analogy is imperfect because gravitational mass and inertia cancel in Newtonian motion, but as a basketball model it could still make sense: some defenders may stay more attached to scheme while others react more aggressively to offensive threats.

A practical version might estimate defender-specific responsiveness or stiffness rather than calling it mass.

## General Goal

### Court Curvature Analogy

Treat offensive players as creating a court-level influence field, and model defenders as moving through that field.

In this framing:

- offensive player gravity creates "curvature" or pressure on the court
- ball/player position and momentum shape the field
- defenders move according to the resulting local defensive incentives

This more of a conceptual guide than a concrete implementation idea. 

### Ultimate Direction

Build a model of how defense reacts to offensive action.

Given the tracking data, players on the floor, ball location, and offensive movement, the goal is to estimate how much each offensive player changes defensive positioning. The current gravity model is a first version of that idea: baseline defensive positioning plus offensive influence.
