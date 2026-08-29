import numpy as np

from scipy.optimize import minimize

from src.event import DEFAULT_EPS

class Solver:
    def __init__(self, season):
        self.season = season

    def solve_mass(self, rim_weight=0.2, p=2, alpha=1e-3, eps=DEFAULT_EPS):
        AtA, Atb, btb, player_list = self.season.normal_equation_terms(rim_weight, p, eps)
        I = np.eye(AtA.shape[0])
        m = np.linalg.solve(AtA + alpha * I, Atb)
        loss = btb - 2 * m @ Atb + m @ AtA @ m + alpha * np.dot(m, m)
        return m, player_list, loss

    def optimize_geometry(self, mass, player_list, current_rim_weight, current_p):
        def objective(x):
            r, p = x
            grav = {pl : m for pl, m in zip(player_list, mass)}
            ls = self.season.loss(grav, r, p)
            return ls
        x0 = (current_rim_weight, current_p)
        return minimize(objective, x0)

    def _constrain_within_bounds(self, val, bounds):
        if val < bounds[0]:
            return bounds[0]
        if val > bounds[1]:
            return bounds[1]
        return val

    def fit_alternating(
        self, 
        initial_rim_weight=0.2, 
        initial_p=2, 
        alpha=1e-3, 
        max_iterations=20, 
        tolerance=1e-3, 
        rim_weight_bounds=(0.0, 0.8), 
        p_bounds=(0.5, 4.0)
    ):
        rim_weight, p = initial_rim_weight, initial_p
        mass, lst, loss = self.solve_mass(rim_weight, p, alpha)
        for _ in range(1, max_iterations):
            # update non-gravity params
            new_weight, new_p = self.optimize_geometry(mass, lst, rim_weight, p)
            new_weight = self._constrain_within_bounds(new_weight, rim_weight_bounds)
            new_p = self._constrain_within_bounds(new_p, p_bounds)
            # update player gravities
            new_mass, new_lst, new_loss = self.solve_mass(new_weight, new_p, alpha)
            # check tolerance
            mass_change = np.abs(new_mass - mass)
            param_change = max(abs(new_weight - rim_weight), abs(new_p - p))
            mass, lst, loss, rim_weight, p = new_mass, new_lst, new_loss, new_weight, new_p
            if mass_change < tolerance and param_change < tolerance:
                return mass, lst, loss, rim_weight, p
        return mass, lst, loss, rim_weight, p