"""Base rules: weighted criteria, then a switching cost (Viterbi). No learning.

Each criterion gives a vote from 0 to 1 for H, A or D on each frame.
The model multiplies each vote by the weight of its criterion and adds them.
A change of label costs points. The model keeps the label path with the best total.
Thus possession changes only when the other side wins clearly for some time.
If no criterion votes, the label does not change (a loose ball stays with the last team).

Criteria (weight):
  reach   (1.0)  The nearest player of each team, near the ball.
  gap     (1.0)  No ball for a long time: dead ball.
  out     (1.5)  Ball over a line, or both teams almost stop: dead ball.
  restart (2.0)  Smart Tagging set piece: dead ball up to the restart, then the taking team.
  pass2   (0.5)  A kick, then 5 m or more of travel: the ball is the passer's in flight.

The weights come from Metrica's two public sample matches, not from the public GT.
Uses only columns of the per-frame feature table (src/features.py, docs/features.md).
"""

import numpy as np
import pandas as pd

FPS = 25
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

WEIGHTS = {"reach": 1.0, "gap": 1.0, "out": 1.5, "restart": 2.0, "pass2": 0.5}
SWITCH_COST = 12.0      # H <-> A. About 0.5 s of clear votes for the other team.
DEAD_COST = 8.0         # into or out of D.

REACH_M = 2.0           # reach: a vote of 0.6 at 2 m, 0.14 at 4 m.
OFFCAM_FACTOR = 0.5     # reach: a player off camera counts half (the tracker guesses him).
GAP_FULL_S = 5.0        # gap: a gap of 5 s or more gives a full vote.
SLOW_TEAM_MS = 1.2      # out: both teams slower than this on average means a stoppage.
SLOW_BALL_MS = 1.0      # restart: a ball slower than this is not in play.
LOOKBACK_S = 30         # restart: the dead ball starts at most 30 s before the restart.
MIN_DEAD_S = 2          # restart: always 2 s of dead ball before the restart.
HOLD_S = 2              # restart: the taking team gets a vote for 2 s.
GOAL_MAX_DEAD_S = 120   # restart: after a goal, dead ball until the kick-off (120 s max).
KICK_WIN = 6            # pass2: velocity before and after a kick, over 6 frames (0.24 s).
KICK_DV_MS = 5.0        # pass2: the velocity must change by 5 m/s or more.
KICK_REACH_M = 2.5      # pass2: the kicker is 2.5 m or nearer to the ball ...
KICK_MARGIN_M = 0.5     # pass2: ... and 0.5 m nearer than all opponents.
FLIGHT_MIN_SPEED_MS = 6.0   # pass2: the ball leaves at 6 m/s or more.
FLIGHT_END_SPEED_MS = 3.0   # pass2: the flight stops below 3 m/s ...
FLIGHT_CLEAR_M = 1.0        # ... or when a player is 1 m or nearer ...
FLIGHT_MAX_S = 4            # ... or after 4 s.
PASS_MIN_LEN_M, PASS_MIN_S = 5.0, 0.4   # pass2: a pass travels 5 m or more, for 0.4 s or more.
FLIGHT_DAMP = 0.2       # pass2: during a pass, reach votes count 0.2 only.


def _col(df, name, fill=np.nan):
    """One feature column as a float array. Missing values become `fill`."""
    return df[name].to_numpy(dtype=float, na_value=fill)


def _runs(mask):
    """Start and end (end not included) of each run of True values."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[0::2], edges[1::2]))


def _passes(df, dist):
    """Per frame: the team of the passer during a pass flight, else -1."""
    n, w = len(df), KICK_WIN
    x, y = _col(df, "ball_x_s"), _col(df, "ball_y_s")
    speed = np.nan_to_num(_col(df, "ball_speed"))
    known = np.isfinite(x)
    # Velocity before and after each frame, each one over 6 frames (this removes the wobble).
    vb = np.full((n, 2), np.nan)
    va = np.full((n, 2), np.nan)
    vb[w:] = np.stack([x[w:] - x[:-w], y[w:] - y[:-w]], 1) * FPS / w
    va[:-w] = np.stack([x[w:] - x[:-w], y[w:] - y[:-w]], 1) * FPS / w
    jump = np.nan_to_num(np.hypot(*(va - vb).T))
    leave = np.nan_to_num(np.hypot(*va.T))
    peak = pd.Series(jump).rolling(2 * w + 1, center=True, min_periods=1).max().to_numpy()
    kicks = np.flatnonzero((jump >= KICK_DV_MS) & (jump >= peak) & (leave >= FLIGHT_MIN_SPEED_MS))
    # Distance of each team to the ball: the median over 5 frames (one bad frame does not count).
    dh = np.nan_to_num(pd.Series(dist[H]).rolling(5, center=True, min_periods=1).median().to_numpy(), nan=np.inf)
    da = np.nan_to_num(pd.Series(dist[A]).rolling(5, center=True, min_periods=1).median().to_numpy(), nan=np.inf)
    near = np.fmin(np.nan_to_num(dist[H], nan=np.inf), np.nan_to_num(dist[A], nan=np.inf))
    passer = np.full(n, -1, dtype=np.int8)
    for f in kicks:
        if dh[f] <= KICK_REACH_M and dh[f] + KICK_MARGIN_M < da[f]:
            team = H
        elif da[f] <= KICK_REACH_M and da[f] + KICK_MARGIN_M < dh[f]:
            team = A
        else:
            continue                      # no clear kicker: not a pass
        g, stop = f + 2, min(n, f + FLIGHT_MAX_S * FPS)
        while g < stop and known[g] and speed[g] >= FLIGHT_END_SPEED_MS and near[g] > FLIGHT_CLEAR_M:
            g += 1                        # the flight continues
        length = np.hypot(x[g - 1] - x[f], y[g - 1] - y[f])
        if g - f >= PASS_MIN_S * FPS and np.isfinite(length) and length >= PASS_MIN_LEN_M:
            passer[f:g] = team
    return passer


def _restart_votes(df, known, out, slow_teams):
    """Votes of the restart criterion, from the Smart Tagging columns of the feature table."""
    n = len(df)
    votes = np.zeros((n, 3))
    speed = np.nan_to_num(_col(df, "ball_speed"), nan=0.0)
    dead_like = ~known | out | (speed < SLOW_BALL_MS) | slow_teams
    live = known & ~out & (speed >= SLOW_BALL_MS)
    side = df["st_set_pieces_side"].astype(str).to_numpy()
    kind = df["st_set_piece_type"].astype(str).to_numpy()
    kickoffs = []
    for a, b in _runs(df["st_set_pieces"].to_numpy(dtype=bool)):
        # The restart: the first frame of 5 frames in a row with the ball seen, on the pitch and moving.
        hits = np.flatnonzero(np.convolve(live[a:b].astype(int), np.ones(5, int), mode="valid") == 5)
        r = a + int(hits[0]) if len(hits) else a
        if kind[a].startswith("kick_off"):
            kickoffs.append(r)
        # Dead ball before the restart: go back while the ball is lost, out or still (30 s max).
        k, floor = max(r - MIN_DEAD_S * FPS, 0), max(r - LOOKBACK_S * FPS, 0)
        while k > floor and dead_like[k - 1]:
            k -= 1
        votes[k:r, D] = 1.0
        if side[a] in ("H", "A"):         # the team in the tag takes the set piece
            votes[r:min(r + HOLD_S * FPS, n), H if side[a] == "H" else A] = 0.5
    # After a goal: dead ball until the next kick-off.
    for _, end in _runs(df["st_goals"].to_numpy(dtype=bool)):
        nxt = [r for r in kickoffs if r >= end]
        stop = min(nxt[0] if nxt else end, end + GOAL_MAX_DEAD_S * FPS)
        votes[end:stop, D] = 1.0
    return votes


def _decode(score, halves):
    """Best H/A/D path in each half (Viterbi) under the switching costs. D outside the halves."""
    sw, dc = SWITCH_COST, DEAD_COST
    out = np.full(len(score), D, dtype=np.int8)
    for a, b in halves:
        s = score[a:b].tolist()
        back = bytearray(3 * len(s))
        vh, va, vd = s[0]
        for i in range(1, len(s)):
            # For each label: the best previous label, after its switching cost.
            bh, ph = (vh, 0) if vh >= va - sw else (va - sw, 1)
            if vd - dc > bh:
                bh, ph = vd - dc, 2
            ba, pa = (va, 1) if va >= vh - sw else (vh - sw, 0)
            if vd - dc > ba:
                ba, pa = vd - dc, 2
            bd, pd_ = (vd, 2) if vd >= vh - dc else (vh - dc, 0)
            if va - dc > bd:
                bd, pd_ = va - dc, 1
            back[3 * i], back[3 * i + 1], back[3 * i + 2] = ph, pa, pd_
            sh, sa, sd = s[i]
            vh, va, vd = bh + sh, ba + sa, bd + sd
        last = 0 if vh >= va and vh >= vd else (1 if va >= vd else 2)
        for i in range(len(s) - 1, -1, -1):   # go back along the best path
            out[a + i] = last
            last = back[3 * i + last]
    return out


def _near_line_gaps(known, out, edge, near_m):
    """True on the frames of each ball gap that began near a line: the last ball seen before the gap
    was `near_m` metres or less from a touchline or goal line (or beyond it), or the ball was seen out
    in the second before the gap."""
    keep = np.zeros(len(known), bool)
    for a, b in _runs(~known):
        if a == 0:
            continue
        near = np.isfinite(edge[a - 1]) and edge[a - 1] <= near_m
        if near or out[max(a - FPS, 0):a].any():
            keep[a:b] = True
    return keep


def predict(data_frames, weights=None, gap_near_line_m=None):
    """Labels H/A/D per frame. weights: the criteria weights (default WEIGHTS above).
    gap_near_line_m: if set, the gap vote only counts for gaps that began this near a line;
    by default every gap counts."""
    W = WEIGHTS if weights is None else weights
    df = data_frames
    n = len(df)
    known = np.isfinite(_col(df, "ball_x_s"))
    out = _col(df, "ball_out", 0) > 0
    dist = np.stack([_col(df, "h_nearest_dist"), _col(df, "a_nearest_dist")])
    offcam = np.stack([_col(df, "h_nearest_offcam", 0) > 0, _col(df, "a_nearest_offcam", 0) > 0])
    team_speed = np.fmax(_col(df, "h_mean_speed"), _col(df, "a_mean_speed"))
    slow_teams = np.nan_to_num(team_speed, nan=np.inf) < SLOW_TEAM_MS
    passer = _passes(df, dist)

    score = np.zeros((n, 3))
    # reach: the nearest player of each team. Less during a pass flight.
    for t in (H, A):
        vote = np.nan_to_num(np.exp(-0.5 * (dist[t] / REACH_M) ** 2))
        vote[offcam[t]] *= OFFCAM_FACTOR
        vote[passer >= 0] *= FLIGHT_DAMP
        score[:, t] += W["reach"] * vote
    # gap: the longer the ball is lost, the more it is a dead ball.
    gap_len = _col(df, "ball_gap_len", 0)
    gap_vote = np.where(known, 0.0, np.clip(gap_len / (GAP_FULL_S * FPS), 0, 1))
    if gap_near_line_m is not None:
        gap_vote[~_near_line_gaps(known, out, _col(df, "ball_edge_dist"), gap_near_line_m)] = 0.0
    score[:, D] += W["gap"] * gap_vote
    # out: ball over a line (full vote), or both teams almost stop (a third of a vote).
    score[:, D] += W["out"] * np.where(out, 1.0, np.where(slow_teams, 1 / 3, 0.0))
    # restart: Smart Tagging set pieces and goals.
    score += W["restart"] * _restart_votes(df, known, out, slow_teams)
    # pass2: during a pass flight, the passer's team.
    for t in (H, A):
        score[:, t] += W["pass2"] * (passer == t)

    halves = _runs(df["half"].to_numpy() > 0)
    return LETTERS[_decode(score, halves)]
