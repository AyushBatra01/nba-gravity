import numpy as np

SCALE = 1
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
BALL_TEAM_ID = -1


class Player:
    """Small wrapper around one location entry from the tracking data."""

    def __init__(self, entry):
        self.team_id = entry[0]
        self.player_id = entry[1]
        # Store coordinates as floats so downstream linear algebra is
        # predictable even if the raw JSON happens to contain ints.
        self.xy = SCALE * np.array([entry[2], entry[3]], dtype=float)

    def to_list(self):
        return [self.team_id, self.player_id, self.xy]

    def __str__(self):
        return f"team_id={self.team_id} player_id={self.player_id} xy={self.xy}"



class Moment:
    """Wrapper around one raw SportVU-style moment."""

    def __init__(self, moment):
        self.id = moment[0]
        self.game_clock = moment[2]
        self.shot_clock = moment[3]
        self.locations = [Player(entry) for entry in moment[5]]

    def location_list(self):
        return [player.to_list() for player in self.locations]

    def rim_location(self):
        # Infer which basket is being attacked from the average non-ball player location.
        player_x = [
            player.xy[0]
            for player in self.locations
            if player.team_id != BALL_TEAM_ID
        ]

        if np.mean(player_x) > SCALE * COURT_LENGTH / 2:
            rim_x = SCALE * COURT_LENGTH
        else:
            rim_x = 0

        return np.array([rim_x, SCALE * COURT_WIDTH / 2])

    def ball_location(self):
        # In this tracking format, the ball is the first location entry.
        return self.locations[0].xy

    def __str__(self):
        return f"id={self.id} clock={self.game_clock} shotclock={self.shot_clock}"




