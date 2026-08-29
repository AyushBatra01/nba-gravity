import numpy as np

SCALE = 1
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0


class Player:
    def __init__(self, entry):
        self.team_id = entry[0]
        self.player_id = entry[1]
        self.xy = SCALE * np.array([entry[2], entry[3]])

    def to_list(self):
        return [self.team_id, self.player_id, self.xy]

    def __str__(self):
        return f"team_id={self.team_id} player_id={self.player_id} xy={self.xy}"



class Moment:
    def __init__(self, moment):
        self.id = moment[0]
        self.game_clock = moment[2]
        self.shot_clock = moment[3]
        self.locations = []
        for entry in moment[5]:
            self.locations.append(Player(entry))

    def location_list(self):
        return [e.to_list() for e in self.locations]

    def rim_location(self):
        if np.mean([e.xy[0] for e in self.locations]) > SCALE * COURT_LENGTH / 2:
            rim_x = SCALE * COURT_LENGTH 
        else:
            rim_x = 0
        return np.array([rim_x, SCALE * COURT_WIDTH / 2])

    def ball_location(self):
        return self.locations[0].xy

    def __str__(self):
        return f"id={self.id} clock={self.game_clock} shotclock={self.shot_clock}"





