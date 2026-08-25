import numpy as np
from itertools import permutations

from util import Moment, Player, SCALE

DEFAULT_EPS = 0.5 * SCALE


class ProcessedMoment:
    def __init__(self, moment, poss_team_id, id=None):
        self.id = id
        self.poss_team_id = poss_team_id
        self.moment = moment

        self.off_players = []
        self.def_players = []
        self.ball = self.moment.ball_location()
        self.rim = self.moment.rim_location()
        for entry in self.moment.locations:
            if entry.team_id == -1:
                continue
            if entry.team_id == poss_team_id:
                self.off_players.append(entry)
            else:
                self.def_players.append(entry)

        self.off_loc = {oplayer.player_id : oplayer.xy for oplayer in self.off_players}
        self.off_loc[-1] = self.ball
        self.off_loc[-2] = self.rim

        self.def_loc = {dplayer.player_id : dplayer.xy for dplayer in self.def_players}

        self.assignments()

    
    # def distance_matrix(self):
    #     # dist and disp matrices using REAL VALUES
    #     dist = {}
    #     disp = {}
    #     for dplayer in self.def_players:
    #         def_id = dplayer.player_id
    #         disp[def_id] = {}
    #         for oplayer in self.off_players:
    #             off_id = oplayer.player_id
    #             disp[def_id][off_id] = oplayer.xy - dplayer.xy
    #         # ball
    #         bxy = self.ball
    #         disp[def_id][-1] = bxy - dplayer.xy
    #         # rim
    #         rxy = self.rim
    #         disp[def_id][-2] = rxy - dplayer.xy
    #         # dists
    #         dist[def_id] = {}
    #         for off_id in disp[def_id].keys():
    #             dist[def_id][off_id] = np.linalg.norm(disp[def_id][off_id])
    #     self.dist_matrix = dist
    #     self.disp_matrix = disp
    #     return dist, disp


    def assignments(self):
        def_pos = np.array([dplayer.xy for dplayer in self.def_players])
        off_pos = np.array([oplayer.xy for oplayer in self.off_players])
        best_perm = None
        best_cost = np.inf
        for perm in permutations(range(5)):
            cost = 0
            for d in range(5):
                o = perm[d]
                cost += np.sum((def_pos[d] - off_pos[o])**2)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        self.matchups = list(best_perm)
        return self.matchups


    def baseline_locations(self, rim_weight=0.2):
        baselines = {}
        matchups = self.matchups
        for i in range(5):
            def_id = self.def_players[i].player_id
            off_matchup = self.off_players[matchups[i]]
            mxy = off_matchup.xy
            rxy = self.rim
            baselines[def_id] = (1 - rim_weight) * mxy + rim_weight * rxy
        return baselines


    def predicted_locations(self, gravity, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        predicted = {}
        base = self.baseline_locations(rim_weight)
        for dplayer in self.def_players:
            def_id = dplayer.player_id
            loc = base[def_id].copy()
            # offensive player + ball + rim effects
            for off_id, off_xy in self.off_loc.items():
                disp = off_xy - base[def_id]
                weight = gravity[off_id] / np.power(np.linalg.norm(disp) + eps, p)
                loc += weight * disp
            predicted[def_id] = loc
        return predicted

    # def actual_locations(self):
    #     actual = {}
    #     for dplayer in self.def_players:
    #         def_id = dplayer.player_id
    #         actual[def_id] = dplayer.xy
    #     return actual


    def resid(self, gravity, rim_weight=0.2, p=2):
        # real = self.actual_locations()
        real = self.def_loc
        pred = self.predicted_locations(gravity, rim_weight, p)
        res = {}
        for def_id in real.keys():
            res[def_id] = real[def_id] - pred[def_id]
        return res


    def loss(self, gravity, rim_weight=0.2, p=2):
        resid = self.resid(gravity, rim_weight, p)
        total = 0
        for r in resid.values():
            total += np.dot(r, r)
        return total

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        real = self.actual_locations()
        base = self.baseline_locations(rim_weight)
        keys = list(real.keys())
        b = [real[k] - base[k] for k in keys]
        A = []
        for def_id in keys:
            coef = {}
            for off_id, off_xy in self.off_loc.items():
                disp = off_xy - base[def_id]
                coef[off_id] = disp / np.power(np.linalg.norm(disp) + eps, p)
            A.append(coef.copy())
        player_set = set(self.off_loc.keys())
        return A, b, player_set

    def normal_equation_terms(self, id_to_index, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        # compute important locations
        def_xy = np.array([player.xy for player in self.def_players])
        matchup_xy = np.array([self.off_players[self.matchups[i]].xy for i in range(5)])
        rim_xy = np.broadcast_to(self.rim, def_xy.shape).copy()
        base = (1 - rim_weight) * matchup_xy + rim_weight * rim_xy
        # target
        b = def_xy - base
        # coefficients
        off_ids = [player.player_id for player in self.off_players] + [-1, -2]
        off_xy = np.array([self.off_loc[pid] for pid in off_ids])
        disp = off_xy[None, :, :] - base[:, None, :]
        dist = np.linalg.norm(disp, axis=2)
        coef = disp / (dist[..., None] ** p + eps)
        # get local contribution
        Ax = coef[:, :, 0]
        Ay = coef[:, :, 1]
        AtA_local = Ax.T @ Ax + Ay.T @ Ay
        Atb_local = Ax.T @ b[:, 0] + Ay.T @ b[:, 1]
        btb = np.sum(b**2)
        # add local contribution into global matrix
        AtA = np.zeros((len(id_to_index), len(id_to_index)))
        Atb = np.zeros(len(id_to_index))
        indices = [id_to_index[pid] for pid in off_ids]
        AtA[np.ix_(indices, indices)] = AtA_local
        Atb[indices] = Atb_local
        return AtA, Atb, btb

    
        
                













        
        
