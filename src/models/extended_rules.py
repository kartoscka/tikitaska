"""Extended rules: the base rules (base_rules.py) with five changes, all of them here. No learning.

base_rules.py is only imported (its helpers and constants), never changed.

1. During a pass flight the reach criterion votes nothing (base rules: x0.2). Only pass2 votes: the
   passer's team.
2. The gap vote counts only for gaps that begin within 1 m of a line.
3. New criterion `phase`: Smart Tagging phases of play vote for the team with the ball.
   The weight is set per Smart Tagging code in PHASE_WEIGHTS.
4. The restart criterion has its own weight per set-piece type (RESTART_TYPE_WEIGHTS) and for goals.
5. Gap: if the ball is lost and the average speed of all players is below SLOW_ALL_MS, the gap votes
   a full dead ball (no length ramp, no near-line condition).
"""

import numpy as np

from src.models import base_rules as _core
from src.models.base_rules import (_col, _runs, _passes, _decode,
                                   _near_line_gaps, H, A, D, LETTERS, FPS)

WEIGHTS = dict(_core.WEIGHTS)
NEAR_LINE_M = 1.0       # a gap that begins this near a line (or beyond it) still votes dead ball
FLIGHT_DAMP = 0.0       # reach votes during a pass flight
SLOW_ALL_MS = 0.5       # gap: ball lost and all players (both teams) slower than this on average = dead ball

# restart: a multiplier per set-piece type (the st_set_piece_type value), on top of WEIGHTS["restart"].
# 1.0 = as the base rules. 0.0 = the type is ignored. "MIX" = several types overlap. A type not listed counts 1.0.
RESTART_TYPE_WEIGHTS = {
    "corner_kick":         1.5,
    "goal_kick":           1.5,
    "throw_in":            1.0,
    "free_kick":           1.5,
    "kick_off_start":      1.5,
    "kick_off_after_goal": 1.5,
    "MIX":                 0.5,
}
GOAL_WEIGHT = 2.0       # multiplier of the dead ball after a goal (until the kick-off)

# phase: one entry per Smart Tagging code (the st_{code} slug): (role, weight).
#   "attack": the Team tag is the team with the ball, so it votes for that team.
#   "defend": the Team tag is the defending team, so it votes for the other team.
# Weight 0.0 = the code is not used. Several codes can be active in the same frame: for each team
# the highest weight counts (a mirrored pair, such as a build up and the other team's defending in the
# attacking third, says the same thing once).
# Starting values: small, because BAR-ATH's phase tags last 15-30 s and cross changes of possession,
# and their real error rate is unknown (0.25 stops paying off when about 30% of the tags point to
# the wrong team). The transitions are lower: the ball just changed team.
PHASE_WEIGHTS = {
    "build_up":                     ("attack", 0.25),
    "progression":                  ("attack", 0.25),
    "creating_chances":             ("attack", 0.25),
    "attacking_transition":         ("attack", 0.125),
    "defending_in_attacking_third": ("defend", 0.25),
    "defending_in_middle_third":    ("defend", 0.25),
    "defending_in_defensive_third": ("defend", 0.25),
    "defensive_transition":         ("defend", 0.125),
}


def _restart_votes(df, known, out, slow_teams, type_weights, goal_weight):
    """The base restart votes (base_rules._restart_votes), each set piece scaled by the weight of its
    type, the dead ball after a goal by goal_weight. With all weights 1.0 the result is the base one."""
    n = len(df)
    votes = np.zeros((n, 3))
    speed = np.nan_to_num(_col(df, "ball_speed"), nan=0.0)
    dead_like = ~known | out | (speed < _core.SLOW_BALL_MS) | slow_teams
    live = known & ~out & (speed >= _core.SLOW_BALL_MS)
    side = df["st_set_pieces_side"].astype(str).to_numpy()
    kind = df["st_set_piece_type"].astype(str).to_numpy()
    kickoffs = []
    for a, b in _runs(df["st_set_pieces"].to_numpy(dtype=bool)):
        # The restart: the first frame of 5 frames in a row with the ball seen, on the pitch and moving.
        hits = np.flatnonzero(np.convolve(live[a:b].astype(int), np.ones(5, int), mode="valid") == 5)
        r = a + int(hits[0]) if len(hits) else a
        if kind[a].startswith("kick_off"):
            kickoffs.append(r)
        w = type_weights.get(kind[a], 1.0)
        # Dead ball before the restart: go back while the ball is lost, out or still (30 s max).
        k, floor = max(r - _core.MIN_DEAD_S * FPS, 0), max(r - _core.LOOKBACK_S * FPS, 0)
        while k > floor and dead_like[k - 1]:
            k -= 1
        votes[k:r, D] = w
        if side[a] in ("H", "A"):         # the team in the tag takes the set piece
            votes[r:min(r + _core.HOLD_S * FPS, n), H if side[a] == "H" else A] = 0.5 * w
    # After a goal: dead ball until the next kick-off.
    for _, end in _runs(df["st_goals"].to_numpy(dtype=bool)):
        nxt = [r for r in kickoffs if r >= end]
        stop = min(nxt[0] if nxt else end, end + _core.GOAL_MAX_DEAD_S * FPS)
        votes[end:stop, D] = goal_weight
    return votes


def _phase_votes(df, phase_weights):
    """(n, 3) votes: H and A from the Smart Tagging phases, nothing for D."""
    votes = np.zeros((len(df), 3))
    for code, (role, weight) in phase_weights.items():
        if weight <= 0 or "st_" + code not in df.columns or "st_" + code + "_side" not in df.columns:
            continue
        active = df["st_" + code].to_numpy(dtype=bool)
        side = df["st_" + code + "_side"].astype(str).to_numpy()
        for team, tag in ((H, "H"), (A, "A")):
            wanted = tag if role == "attack" else ("A" if tag == "H" else "H")
            hit = active & (side == wanted)
            votes[hit, team] = np.maximum(votes[hit, team], weight)
    return votes


def predict(data_frames, weights=None, phase_weights=None, restart_type_weights=None, goal_weight=None,
            slow_all_ms=None):
    """Labels H/A/D per frame. weights: the criteria weights (default WEIGHTS above).
    phase_weights: the weight of each Smart Tagging code (default PHASE_WEIGHTS above).
    restart_type_weights, goal_weight: the restart multipliers (defaults RESTART_TYPE_WEIGHTS, GOAL_WEIGHT).
    slow_all_ms: the "very slow" speed of the gap rule (default SLOW_ALL_MS; 0 turns the rule off)."""
    W = WEIGHTS if weights is None else weights
    SLOW = SLOW_ALL_MS if slow_all_ms is None else slow_all_ms
    PW = PHASE_WEIGHTS if phase_weights is None else phase_weights
    RW = RESTART_TYPE_WEIGHTS if restart_type_weights is None else restart_type_weights
    GW = GOAL_WEIGHT if goal_weight is None else goal_weight
    df = data_frames
    n = len(df)
    known = np.isfinite(_col(df, "ball_x_s"))
    out = _col(df, "ball_out", 0) > 0
    dist = np.stack([_col(df, "h_nearest_dist"), _col(df, "a_nearest_dist")])
    offcam = np.stack([_col(df, "h_nearest_offcam", 0) > 0, _col(df, "a_nearest_offcam", 0) > 0])
    team_speed = np.fmax(_col(df, "h_mean_speed"), _col(df, "a_mean_speed"))
    slow_teams = np.nan_to_num(team_speed, nan=np.inf) < _core.SLOW_TEAM_MS
    passer = _passes(df, dist)

    score = np.zeros((n, 3))
    # reach: the nearest player of each team. FLIGHT_DAMP during a pass flight.
    for t in (H, A):
        vote = np.nan_to_num(np.exp(-0.5 * (dist[t] / _core.REACH_M) ** 2))
        vote[offcam[t]] *= _core.OFFCAM_FACTOR
        vote[passer >= 0] *= FLIGHT_DAMP
        score[:, t] += W["reach"] * vote
    # gap: the longer the ball is lost, the more it is a dead ball. Only near a line.
    gap_len = _col(df, "ball_gap_len", 0)
    gap_vote = np.where(known, 0.0, np.clip(gap_len / (_core.GAP_FULL_S * FPS), 0, 1))
    gap_vote[~_near_line_gaps(known, out, _col(df, "ball_edge_dist"), NEAR_LINE_M)] = 0.0
    # Very slow rule: ball lost and the average speed of all players below SLOW = a full dead ball.
    all_speed = np.nanmean(np.stack([_col(df, "h_mean_speed"), _col(df, "a_mean_speed")]), axis=0)
    gap_vote[~known & (np.nan_to_num(all_speed, nan=np.inf) < SLOW)] = 1.0
    score[:, D] += W["gap"] * gap_vote
    # out: ball over a line (full vote), or both teams almost stop (a third of a vote).
    score[:, D] += W["out"] * np.where(out, 1.0, np.where(slow_teams, 1 / 3, 0.0))
    # restart: Smart Tagging set pieces and goals, scaled per set-piece type.
    score += W["restart"] * _restart_votes(df, known, out, slow_teams, RW, GW)
    # pass2: during a pass flight, the passer's team.
    for t in (H, A):
        score[:, t] += W["pass2"] * (passer == t)
    # phase: Smart Tagging phases of play. The weight is already in the votes.
    score += _phase_votes(df, PW)

    halves = _runs(df["half"].to_numpy() > 0)
    return LETTERS[_decode(score, halves)]
