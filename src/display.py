import matplotlib.pyplot as plt

from src.util import BALL_TEAM_ID, COURT_LENGTH, COURT_WIDTH, SCALE


def visualize_moment(moment, poss_team, ax=None):
    """Plot one tracking moment with offense, defense, ball, and rim.

    Passing an axis makes this easier to use inside notebooks with subplots.
    If no axis is provided, the function creates a normal pyplot figure.
    """
    off_x, off_y, def_x, def_y = [], [], [], []

    for player in moment.locations:
        if player.team_id == BALL_TEAM_ID:
            continue

        if player.team_id == poss_team:
            off_x.append(player.xy[0])
            off_y.append(player.xy[1])
        else:
            def_x.append(player.xy[0])
            def_y.append(player.xy[1])
    rim_xy = moment.rim_location()
    ball_xy = moment.ball_location()

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(off_x, off_y, label="Offense")
    ax.scatter(def_x, def_y, label="Defense")
    ax.scatter([ball_xy[0]], [ball_xy[1]], label="Ball")
    ax.scatter([rim_xy[0]], [rim_xy[1]], label="Rim")
    ax.set_xlim(0, SCALE * COURT_LENGTH)
    ax.set_ylim(0, SCALE * COURT_WIDTH)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    return ax
