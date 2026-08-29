import matplotlib.pyplot as plt

from src.util import Player, Moment, SCALE, COURT_LENGTH, COURT_WIDTH
    

def visualize_moment(moment, poss_team):
    locs = moment.locations
    off_x, off_y, def_x, def_y = [], [], [], []
    for player in locs:
        if player.team_id == -1:
            continue
        if player.team_id == poss_team:
            off_x.append(player.xy[0])
            off_y.append(player.xy[1])
        else:
            def_x.append(player.xy[0])
            def_y.append(player.xy[1])
    
    rxy = moment.rim_location()
    bxy = moment.ball_location()

    plt.scatter(off_x, off_y, label="Offense")
    plt.scatter(def_x, def_y, label="Defense")
    plt.scatter([bxy[0]], [bxy[1]], label="Ball")
    plt.scatter([rxy[0]], [rxy[1]], label="Rim")
    plt.xlim(0,SCALE * COURT_LENGTH)
    plt.ylim(0,SCALE * COURT_WIDTH)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()
