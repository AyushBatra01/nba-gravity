import numpy as np

from src.util import Player, Moment, SCALE

def get_logs(game):
    # get all logs for an entire game
    pass

def get_event_logs(event, gap=0.25, pad=3.0, bound_start=None, bound_end=None):
    """
    get logs for a single event

    Parameters
    ----------
    event : dict
        dictionary containing an entry for 'moments'

    gap : float
        number of seconds between each logged moment

    pad : float
        number of seconds to cut from event start and end (to avoid transition)

    bound_start : None or float
        only look for moments after this game clock time (not active if None)

    boun_end : None or float
        only look for moments before this game clock time (not active if None)
    """
    log = []
    event_idx = []
    moments = [Moment(m) for m in event['moments']]
    begin = moments[0].game_clock
    end = moments[-1].game_clock
    if bound_start is not None:
        begin = min(begin, bound_start)
    if bound_end is not None:
        end = max(end, bound_end)

    last = begin
    n_error = 0
    for i in range(len(moments)-1):
        moment = moments[i]
        moment_next = moments[i+1]
        clock = moment.game_clock
        clock_next = moment_next.game_clock
        time_elapsed = clock - clock_next
        if time_elapsed != 0 and begin - clock > pad and clock - end > pad and last - clock >= gap:
            if not all_moment_checks(moment, moment_next):
                n_error += 1
                continue
            log.append(moment)
            event_idx.append(i)
            last = clock
    return log, event_idx, n_error


def possession_team(event_logs, check_tie=True):
    """
    find team with possession for the event from the event logs
    determines possessing team based on which team is closest to the ball for the majority of the logged moments

    Parameters
    ----------
    event_logs : lst
        output of get_event_logs()

    check_tie : bool
        checks if both teams have same number of instances where closest to ball
    """
    counts = {}
    totals = {}
    top_n = 0
    poss_tm = None
    for moment in event_logs:
        locs = moment.locations
        ball_xy = moment.ball_location()
        lowest = 9999999
        tm = None
        for i in range(1,11):
            dist = np.linalg.norm(ball_xy - locs[i].xy)
            if dist < lowest:
                lowest = dist
                tm = locs[i].team_id
        if tm not in counts:
            counts[tm] = 0
        counts[tm] += 1
        if tm not in totals:
            totals[tm] = 0.0
        totals[tm] += lowest
        if counts[tm] > top_n:
            top_n = counts[tm]
            poss_tm = tm
    # ensure its not a tie
    if check_tie:
        n_with_top = 0
        for tm, cnt in counts.items():
            if cnt == top_n:
                n_with_top += 1
        # assert n_with_top == 1, "Unable to discern possessing team!"
        low_tm = None
        low_val = 9999999
        for tm, cnt in counts.items():
            if cnt == top_n and totals[tm] < low_val:
                low_tm = tm
                low_val = totals[tm]
        poss_tm = low_tm
    return poss_tm






def check_no_overlap(moment):
    # no completely overlapping players
    locs = set()
    for player in moment.locations:
        if player.player_id in [-1,-2]:
            continue
        pxy = (player.xy[0], player.xy[1])
        if pxy in locs:
            return False
        locs.add(pxy)
    return True

def check_distance(moment1, moment2, max_dist_per_sec=SCALE*50):
    # make sure not unrealistic distance
    clock_gap = np.abs(moment1.game_clock - moment2.game_clock)
    locs1 = {}
    for player in moment1.locations:
        pid = player.player_id
        locs1[pid] = player.xy
    for player in moment2.locations:
        pid = player.player_id
        if pid not in locs1:
            return False
        pxy1 = locs1[pid]
        pxy2 = player.xy
        d = np.linalg.norm(pxy1 - pxy2)
        if d > clock_gap * max_dist_per_sec:
            return False
    return True

def check_players(moment1, moment2):
    # make sure same 10 players
    players1 = set()
    for player in moment1.locations:
        players1.add(player.player_id)
    if len(players1) != 11:
        return False
    for player in moment2.locations:
        if player.player_id not in players1:
            return False
    return True

def all_moment_checks(moment1, moment2):
    return check_no_overlap(moment1) and check_players(moment1, moment2) and check_distance(moment1, moment2)






    