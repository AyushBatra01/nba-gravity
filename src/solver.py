import numpy as np
from scipy.optimize import minimize

from src.model import GravityConfig


class Solver:
    """Higher-level object to compute gravity values and optionally tune geometry parameters."""

    def __init__(self, season):
        self.season = season

    def solve_gravity(self, config=None):
        """Fit gravity values for a fixed model configuration."""
        return self.season.solve_gravity(config)

    def optimize_geometry(
        self,
        gravity_values,
        player_list,
        current_config,
        rim_weight_bounds=(0.0, 0.8),
        decay_power_bounds=(0.5, 4.0),
        max_iterations=25,
    ):
        """Tune rim_weight and decay_power while holding gravity fixed."""
        def objective(x):
            rim_weight, decay_power = x

            # For a candidate geometry, rebuild the normal-equation pieces and
            # score the current gravity vector with matrix math. This is much
            # faster than walking every moment through predicted_locations.
            AtA, Atb, btb, candidate_player_list = self.season.normal_equation_terms(
                rim_weight=rim_weight,
                p=decay_power,
                eps=current_config.eps,
            )

            if candidate_player_list != player_list:
                raise ValueError("Geometry optimization changed the fitted player list.")

            return self.season.loss_from_terms(
                gravity_values,
                AtA,
                Atb,
                btb,
                alpha=current_config.alpha,
            )

        result = minimize(
            objective,
            x0=np.array([current_config.rim_weight, current_config.decay_power]),
            method="L-BFGS-B",
            bounds=[rim_weight_bounds, decay_power_bounds],
            options={"maxiter": max_iterations},
        )

        return result.x[0], result.x[1], result

    def fit_alternating(
        self,
        initial_config=None,
        max_iterations=20,
        tolerance=1e-3,
        rim_weight_bounds=(0.0, 0.8),
        decay_power_bounds=(0.5, 4.0),
        geometry_max_iterations=25,
        verbose=False,
    ):
        """Alternate between fitting gravity values and geometry parameters.

        This is an EM-like coordinate-descent loop:
        1. Fit gravity values with the current geometry.
        2. Hold gravity fixed and optimize the geometry.
        3. Repeat until the fitted values and geometry stop moving much.
        """
        if initial_config is None:
            initial_config = GravityConfig()

        config = initial_config
        gravity_values, player_list, loss = self.solve_gravity(config)

        for iteration in range(max_iterations):
            new_rim_weight, new_decay_power, geometry_result = self.optimize_geometry(
                gravity_values,
                player_list,
                config,
                rim_weight_bounds=rim_weight_bounds,
                decay_power_bounds=decay_power_bounds,
                max_iterations=geometry_max_iterations,
            )

            new_config = GravityConfig(
                # scipy returns numpy scalar types; converting to plain floats
                # keeps printed configs and downstream serialization simple.
                rim_weight=float(new_rim_weight),
                decay_power=float(new_decay_power),
                eps=config.eps,
                alpha=config.alpha,
            )

            new_gravity_values, new_player_list, new_loss = self.solve_gravity(new_config)

            gravity_change = np.max(np.abs(new_gravity_values - gravity_values))
            config_change = max(
                abs(new_config.rim_weight - config.rim_weight),
                abs(new_config.decay_power - config.decay_power),
            )

            gravity_values = new_gravity_values
            player_list = new_player_list
            loss = new_loss
            config = new_config

            if verbose:
                print(
                    f"Iteration {iteration + 1}: "
                    f"loss={loss:.4f}, "
                    f"rim_weight={config.rim_weight:.4f}, "
                    f"decay_power={config.decay_power:.4f}, "
                    f"geometry_success={geometry_result.success}"
                )

            if gravity_change < tolerance and config_change < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations.")
                break

        return gravity_values, player_list, loss, config
