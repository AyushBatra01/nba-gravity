import numpy as np

from process import ProcessedMoment, DEFAULT_EPS
from gather import get_event_logs, possession_team
from util import Moment


MINIMUM_MOMENTS = 5


class Event:
    def __init__(self, event, gap=0.25, pad=3.0, bound_start=None, bound_end=None):
        self.raw = event
        logs, idx, nerr = get_event_logs(event, gap, pad, bound_start, bound_end)
        self.logs = logs
        self.idx = idx
        self.min_moments = MINIMUM_MOMENTS
        self.poss_team = None
        if self.valid_event():
            self.poss_team = possession_team(logs, check_tie=True)

    def valid_event(self):
        return len(self.logs) >= self.min_moments

    def process_moments(self):
        self.moments = [ProcessedMoment(m, self.poss_team, _id) for m, _id in zip(self.logs, self.idx)]

    def loss(self, gravity, rim_weight=0.2, p=2):
        total = 0
        for m in self.moments:
            total += m.loss(gravity, rim_weight, p)
        return total

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        assert self.moments is not None
        A, b = [], []
        player_set = set()
        for m in self.moments:
            Am, bm, players = m.matrix(rim_weight, p, eps)
            A.extend(Am)
            b.extend(bm)
            player_set |= players
        return A, b, player_set



class Game:
    def __init__(self, game, gap=0.25, pad=3.0):
        self.id = game['gameid']
        self.date = game['gamedate']
        self.home = game['events'][0]['home']
        self.away = game['events'][0]['visitor']
        
        self.events = []
        errors = []
        start = None
        for i, event in enumerate(game['events']):
            if len(event['moments']) == 0:
                continue
            if Moment(event['moments'][0]).game_clock == 720.0:
                start = None
            try:
                ev = Event(event, gap, pad, start)
            except:
                errors.append(i)
                continue
            if ev.valid_event():
                ev.process_moments()
                self.events.append(ev)
                start = ev.logs[-1].game_clock
        self.errors = errors

    
    def loss(self, gravity, rim_weight, p):
        total = 0
        for ev in self.events:
            total += ev.loss(gravity, rim_weight, p)
        return total

    def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        A, b = [], []
        player_set = set()
        for e in self.events:
            Ae, be, players = e.matrix(rim_weight, p, eps)
            A.extend(Ae)
            b.extend(be)
            player_set |= players
        return A, b, player_set



class Season:
    def __init__(self, games, gap=0.25, pad=3.0):
        self.games = []
        for g in games:
            self.games.append(Game(g, gap, pad))
        self.A = None
        self.b = None
        # self.player_list = None
        self.player_list = sorted({
            player.player_id
            for game in self.games
            for event in game.events
            for moment in event.moments
            for player in moment.off_players
        } | {-1, -2})
        self.id_to_index = {pid: i for i, pid in enumerate(self.player_list)}


    def loss(self, gravity, rim_weight, p):
        total = 0
        for g in self.games:
            total += g.loss(gravity, rim_weight, p)
        return total

    # def matrix(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
    #     A_raw, b_raw = [], []
    #     player_set = set()
    #     for g in self.games:
    #         Ag, bg, players = g.matrix(rim_weight, p, eps)
    #         A_raw.extend(Ag)
    #         b_raw.extend(bg)
    #         player_set |= players
    #     # convert into correct data structures (once for x coord, once for y coord)
    #     player_list = sorted(list(player_set))
    #     id_to_index = {}
    #     for i, pid in enumerate(player_list):
    #         id_to_index[pid] = i
    #     Ax, Ay = np.zeros(shape=(len(A_raw), len(player_list))), np.zeros(shape=(len(A_raw), len(player_list)))
    #     bx, by = np.zeros(shape=len(b_raw)), np.zeros(shape=len(b_raw))
    #     for i in range(len(A_raw)):
    #         bx[i] = b_raw[i][0]
    #         by[i] = b_raw[i][1]
    #         for off_id, wt in A_raw[i].items():
    #             j = id_to_index[off_id]
    #             Ax[i,j] = A_raw[i][off_id][0]
    #             Ay[i,j] = A_raw[i][off_id][1]
    #     A = np.vstack((Ax, Ay))
    #     b = np.concatenate((bx, by))
    #     self.A = A
    #     self.b = b
    #     self.player_list = player_list
    #     return A, b, player_list

    # def solve_mass(self, rim_weight=0.2, p=2, alpha=1e-3, eps=DEFAULT_EPS):
    #     A, b, player_list = self.matrix(rim_weight, p, eps)
    #     AtA = A.T @ A
    #     Atb = A.T @ b
    #     I = np.eye(AtA.shape[0])
    #     m = np.linalg.solve(AtA + alpha * I, Atb)
    #     return m, player_list


    def normal_equation_terms(self, rim_weight=0.2, p=2, eps=DEFAULT_EPS):
        n_players = len(self.player_list)
        AtA = np.zeros((n_players, n_players))
        Atb = np.zeros(n_players)
        btb = 0
        for game in self.games:
            for event in game.events:
                for moment in event.moments:
                    AtA_m, Atb_m, btb_m = moment.normal_equation_terms(self.id_to_index, rim_weight, p, eps)
                    AtA += AtA_m
                    Atb += Atb_m
                    btb += btb_m
        return AtA, Atb, btb, self.player_list
          

    def solve_mass(self, rim_weight=0.2, p=2, alpha=1e-3, eps=DEFAULT_EPS):
        AtA, Atb, btb, player_list = self.normal_equation_terms(rim_weight, p, eps)
        I = np.eye(AtA.shape[0])
        m = np.linalg.solve(AtA + alpha * I, Atb)
        loss = btb - 2 * m @ Atb + m @ AtA @ m
        return m, player_list, loss


    def fit_alternating(self, initial_rim_weight=0.2, initial_p=2, alpha=1e-3, max_iterations=20, tolerance=1e-3, 
                        rim_weight_bounds=(0.0, 0.8), p_bounds=(0.5, 4.0)):
        rim_weight, p = initial_rim_weight, initial_p
        mass, lst, loss = self.solve_mass(rim_weight, p, alpha)
        for iteration in range(1, max_iterations+1):
            # TO DO!!!
            new_weight, new_p = self.optimize_geometry()
            #
            new_mass, new_lst, new_loss = self.solve_mass(new_weight, new_p, alpha)
            mass_change = np.abs(new_mass - mass)
            param_change = max(abs(new_weight - rim_weight), abs(new_p - p))
            mass, lst, loss, rim_weight, p = new_mass, new_lst, new_loss, new_weight, new_p
            if mass_change < tolerance and param_change < tolerance:
                return mass, lst, loss, rim_weight, p
        return mass, lst, loss, rim_weight, p
            



    