"""Per-frame feature table for the tactical video, plus its data dictionary.

One row per tactical video frame (0 .. n-1, same index as the submission). Pure functions do the
work; the CLI loads the data and writes the table to reports/features.parquet (gitignored), and can
write the data dictionary docs/features.md straight from the column spec in this file, so the two
never drift apart.

Coordinates
-----------
Metres. x = ATD normalised x * pitch length (105.3 m, games.py), y = ATD normalised y * pitch
width (68.0 m). The origin is the pitch corner at ATD (0, 0); x runs along the touchlines from
one goal line to the other, y runs along the goal lines from one touchline to the other.
Which way each team attacks is derived per half (column h_attacks_pos_x).

Constants
---------
Every threshold is physical or football reasoning, written next to it below and copied into the
dictionary. None is fitted on this match, and the public GT is never read.

Output
------
stdout only ever says "features written", "dictionary written" or
"FAILED: <type> at <file>:<line>". The table stays inside reports/ (match data is never committed).

Run (from the repo root):
  python -m src.features
Dictionary only (reads no data):
  python -m src.features --dictionary-only
"""

import argparse
import os
import re
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "starter")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_X, PITCH_Y = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1
TEAMS = (("h", HOME), ("a", AWAY))

REPORT_DIR = os.path.join(ROOT, "reports")
DEFAULT_TABLE = os.path.join("reports", "features.parquet")
DEFAULT_DICTIONARY = os.path.join(ROOT, "docs", "features.md")

# ------------------------------------------------------------------------------------------------
# Constants. Physical / football reasoning only; nothing here is fitted on this match.
# ------------------------------------------------------------------------------------------------

BALL_MAX_SPEED_MS = 50.0
PLAYER_MAX_SPEED_MS = 12.5
BALL_POS_TOL_M = 1.0
PLAYER_POS_TOL_M = 0.5
MIN_PIECE_FRAMES = 5
BALL_INTERP_MAX_FRAMES = 25
BALL_FIT_HALF = 4
PLAYER_FIT_HALF = 6
BALL_TURN_LAG = 4
BALL_TURN_MIN_SPEED_MS = 2.0
BALL_OUT_MARGIN_M = 1.0
CONTROL_RADIUS_M = 2.0
PRESSURE_RADIUS_M = 5.0
IMAGE_SIZE = (1920, 1080)
MAX_CLEAN_PASSES = 50

CONSTANTS = [
    ("BALL_MAX_SPEED_MS", BALL_MAX_SPEED_MS, "m/s",
     "The hardest shots reach about 35-40 m/s. Anything faster between two observations is a "
     "tracker error, not the ball."),
    ("PLAYER_MAX_SPEED_MS", PLAYER_MAX_SPEED_MS, "m/s",
     "Top sprinters peak around 10-11 m/s. Faster means the slot switched to another person or "
     "a false detection."),
    ("BALL_POS_TOL_M", BALL_POS_TOL_M, "m",
     "Slack added to every ball jump test. The ATD ball has no height and its dot sits behind or "
     "beside the real ball by up to about a metre (seen on the video), so that much is noise, not a "
     "jump."),
    ("PLAYER_POS_TOL_M", PLAYER_POS_TOL_M, "m",
     "Slack added to every player jump test: the foot point of a detection box wobbles by about "
     "half a metre when legs or neighbours overlap."),
    ("MIN_PIECE_FRAMES", MIN_PIECE_FRAMES, "frames",
     "0.2 s. A stretch of a track that is reached and left by physically impossible jumps, is "
     "shorter than this, and whose neighbours line up again across it, is a short swap to the "
     "wrong object (a real ball or player cannot teleport away and back). It is dropped."),
    ("BALL_INTERP_MAX_FRAMES", BALL_INTERP_MAX_FRAMES, "frames",
     "1 s. Within about a second a ball in play keeps its path (a pass or a dribble), so a "
     "straight line between the last and next sightings is a fair guess. Longer gaps can hide a "
     "touch, a deflection or a stoppage, so they stay missing."),
    ("BALL_FIT_HALF", BALL_FIT_HALF, "frames",
     "Half-width of the local straight-line fit for ball position and velocity: 9 frames, "
     "0.36 s. Long enough to average the per-frame detection noise, short enough that a pass "
     "(usually longer than 0.3 s) stays a separate movement."),
    ("PLAYER_FIT_HALF", PLAYER_FIT_HALF, "frames",
     "Half-width of the local straight-line fit for player position and velocity: 13 frames, "
     "0.52 s. Players change velocity more slowly than the ball, so a wider window is safe."),
    ("BALL_TURN_LAG", BALL_TURN_LAG, "frames",
     "ball_turn compares the ball velocity 4 frames (0.16 s) before and after a frame: about the "
     "time a touch takes to show up in the fitted velocity."),
    ("BALL_TURN_MIN_SPEED_MS", BALL_TURN_MIN_SPEED_MS, "m/s",
     "Below walking pace the direction of the fitted velocity is mostly noise, so ball_turn is "
     "left empty."),
    ("BALL_OUT_MARGIN_M", BALL_OUT_MARGIN_M, "m",
     "The ball is out only when all of it is over the line (radius 0.11 m) and the ATD position "
     "is noisy by up to about a metre, so the ball must be this far past a line to count as out."),
    ("CONTROL_RADIUS_M", CONTROL_RADIUS_M, "m",
     "A player controlling the ball has it within a stride (about 1-1.5 m); plus position noise."),
    ("PRESSURE_RADIUS_M", PRESSURE_RADIUS_M, "m",
     "Usual distance at which a defender counts as pressing the ball carrier."),
    ("IMAGE_SIZE", IMAGE_SIZE, "pixels",
     "Size of the tactical video the homography maps into. A player "
     "projected outside it is off camera."),
    ("MAX_CLEAN_PASSES", MAX_CLEAN_PASSES, "passes",
     "Safety cap on the jump-removal loop (it normally stops after a few passes). Not a tuning "
     "knob."),
]

KNOWN_ST_CODES = [
    "PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PROGRESSION",
    "DEFENDING IN MIDDLE THIRD", "SET PIECES", "CREATING CHANCES", "DEFENDING IN DEFENSIVE THIRD",
    "LONG BALL", "ATTACKING TRANSITION", "DEFENSIVE TRANSITION", "BUILD UP",
    "DEFENDING IN ATTACKING THIRD", "GOALS",
]
KNOWN_SET_PIECE_TYPES = [
    "Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)",
    "Kick Off (Start)",
]
SIDE_NAMES = np.array(["none", "H", "A", "NA", "MIX"], dtype=object)   # index = side code
SIDE_NONE, SIDE_H, SIDE_A, SIDE_NA, SIDE_MIX = range(5)


def slug(text):
    """'DEFENDING IN MIDDLE THIRD' -> 'defending_in_middle_third'."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


# ------------------------------------------------------------------------------------------------
# Column spec: the single source for the data dictionary.
# {t} = h (home, Barcelona) or a (away, Athletic). {code} = slug of a Smart Tagging code.
# (name, dtype, unit, definition, cleaning, constants)
# ------------------------------------------------------------------------------------------------

COLUMNS = [
    ("frame", "int32", "-", "Tactical video frame, 0-based. Same as the submission row and the "
     "homography row; ATD frame_id - 1.", "-", "-"),
    ("half", "int8", "-", "1 or 2 inside a graded half (games.half_video_blocks), 0 elsewhere "
     "(before the match, halftime, after).", "-", "-"),
    ("graded", "bool", "-", "Frame is scored (inside a half). Note the start of half 2 is graded "
     "before the kick-off.", "-", "-"),
    ("hom_valid", "bool", "-", "Homography row is valid.", "-", "-"),
    ("hom_error", "float32", "px", "Homography reprojection error (lower is better).", "As given.",
     "-"),
    # Ball
    ("ball_x", "float32", "m", "Raw ATD ball x in metres.", "None (raw).", "-"),
    ("ball_y", "float32", "m", "Raw ATD ball y in metres.", "None (raw).", "-"),
    ("ball_missing", "bool", "-", "ATD has no ball on this frame (NaN).", "-", "-"),
    ("ball_jump", "bool", "-", "Ball was present but dropped as a jump (short swap to a wrong "
     "object). The frame may still get an interpolated position (then ball_interp is also True).",
     "Jump removal, see ball_x_c.",
     "BALL_MAX_SPEED_MS, BALL_POS_TOL_M, MIN_PIECE_FRAMES"),
    ("ball_interp", "bool", "-", "ball_x_c / ball_y_c on this frame are a straight-line fill of a "
     "short gap, not an observation.", "See ball_x_c.", "BALL_INTERP_MAX_FRAMES"),
    ("ball_gap_len", "int32", "frames", "Length of the stretch without a usable ball observation "
     "(missing or dropped as a jump) that contains this frame; 0 on observed frames. Counted "
     "before interpolation, so interpolated frames keep their gap length.", "-",
     "MIN_PIECE_FRAMES"),
    ("ball_x_c", "float32", "m", "Cleaned ball x: raw, jumps removed, short gaps filled.",
     "1) Consecutive observations further apart than max speed x elapsed time + tolerance split "
     "the track into pieces. 2) Pieces shorter than the minimum whose neighbours line up across "
     "them are dropped (repeat until stable). 3) Gaps up to the interpolation limit inside one "
     "piece are filled linearly. Longer gaps stay NaN.",
     "BALL_MAX_SPEED_MS, BALL_POS_TOL_M, MIN_PIECE_FRAMES, BALL_INTERP_MAX_FRAMES"),
    ("ball_y_c", "float32", "m", "Cleaned ball y, as ball_x_c.", "As ball_x_c.",
     "as ball_x_c"),
    ("ball_x_s", "float32", "m", "Smoothed ball x: value of a local straight-line fit centred on "
     "the frame (a moving average that keeps constant-speed motion exact). Falls back to "
     "ball_x_c where the window has too few points. NaN where ball_x_c is NaN.",
     "Fit uses only cleaned points of the same piece (never across a jump or a long gap).",
     "BALL_FIT_HALF"),
    ("ball_y_s", "float32", "m", "Smoothed ball y, as ball_x_s.", "As ball_x_s.",
     "BALL_FIT_HALF"),
    ("ball_vx", "float32", "m/s", "Ball velocity x: slope of the local straight-line fit. The ATD "
     "speed_kph column is NOT used (it matches neither km/h nor m/s).",
     "Needs more than half the window filled with cleaned points of the same piece, else NaN.",
     "BALL_FIT_HALF"),
    ("ball_vy", "float32", "m/s", "Ball velocity y, as ball_vx.", "As ball_vx.", "BALL_FIT_HALF"),
    ("ball_speed", "float32", "m/s", "Magnitude of (ball_vx, ball_vy). Horizontal only: the ATD "
     "has no ball height.", "As ball_vx.", "BALL_FIT_HALF"),
    ("ball_accel", "float32", "m/s^2", "Magnitude of the slope of a local straight-line fit of "
     "the ball velocity: how fast the velocity vector changes. Touches, passes and stops show as "
     "peaks.", "Same windowing as ball_vx, applied to the velocity.", "BALL_FIT_HALF"),
    ("ball_turn", "float32", "rad", "Angle between the ball velocity BALL_TURN_LAG frames before "
     "and after (0 = straight on, pi = reversed). NaN when either speed is below the minimum or "
     "the two frames are in different pieces.", "-",
     "BALL_TURN_LAG, BALL_TURN_MIN_SPEED_MS"),
    ("ball_out_x", "bool", "-", "Cleaned ball beyond a goal line (x < 0 or x > length) by more "
     "than the margin. False when the ball is unknown.", "Uses ball_x_c.", "BALL_OUT_MARGIN_M"),
    ("ball_out_y", "bool", "-", "Cleaned ball beyond a touchline (y < 0 or y > width) by more "
     "than the margin. False when the ball is unknown.", "Uses ball_y_c.", "BALL_OUT_MARGIN_M"),
    ("ball_out", "bool", "-", "ball_out_x or ball_out_y.", "-", "BALL_OUT_MARGIN_M"),
    ("ball_edge_dist", "float32", "m", "Signed distance from the cleaned ball to the nearest "
     "pitch line (goal line or touchline): positive inside, negative outside.",
     "Uses ball_x_c, ball_y_c.", "-"),
    ("h_attacks_pos_x", "boolean (nullable)", "-", "Home attacks towards larger x in this half. "
     "Derived per half: the team whose deepest player (smallest x) usually sits nearer x = 0 "
     "defends that goal (its keeper), checked together with the most advanced player (largest "
     "x). Medians over all frames of the half, so a wrong keeper slot on a few frames does not "
     "matter. NA outside the halves.", "Uses cleaned player positions.", "-"),
    ("ball_x_att_h", "float32", "m", "Smoothed ball distance from the home goal line along the "
     "pitch (0 = home goal line, pitch length = away goal line).", "Uses ball_x_s and "
     "h_attacks_pos_x.", "-"),
    # Per team
    ("{t}_n_tracked", "int8", "players", "Slots of this team with a cleaned position on this "
     "frame (can exceed 11 with duplicates).", "Player jump removal, see note.",
     "PLAYER_MAX_SPEED_MS, PLAYER_POS_TOL_M, MIN_PIECE_FRAMES"),
    ("{t}_n_offcam", "int8", "players", "Tracked slots of this team whose position projects "
     "outside the video image through the homography. These are extrapolated by the tracker "
     "and unreliable (a keeper slot can sit on the wrong player).", "-",
     "IMAGE_SIZE"),
    ("{t}_n_dropped", "int8", "players", "Slots of this team whose sample on this frame was "
     "dropped as a jump.", "-", "PLAYER_MAX_SPEED_MS, PLAYER_POS_TOL_M, MIN_PIECE_FRAMES"),
    ("{t}_nearest_dist", "float32", "m", "Distance from ball_x_s, ball_y_s to the nearest "
     "tracked player of this team. NaN when the ball or the team is unknown.",
     "Player position = local fit value (else cleaned position).", "PLAYER_FIT_HALF"),
    ("{t}_nearest_slot", "Int16 (nullable)", "-", "Track slot number (1-based, as 'Track N' in "
     "the XML) of that nearest player. A slot is a tracklet, not a "
     "person.", "-", "-"),
    ("{t}_nearest_offcam", "boolean (nullable)", "-", "That nearest player is off camera "
     "(see {t}_n_offcam).", "-", "IMAGE_SIZE"),
    ("{t}_second_dist", "float32", "m", "Distance to the second nearest tracked player of this "
     "team.", "As {t}_nearest_dist.", "PLAYER_FIT_HALF"),
    ("{t}_n_within_2m", "Int8 (nullable)", "players", "Tracked players of this team within the "
     "control radius of the ball. NA when the ball is unknown.", "-", "CONTROL_RADIUS_M"),
    ("{t}_n_within_5m", "Int8 (nullable)", "players", "Tracked players of this team within the "
     "pressure radius of the ball. NA when the ball is unknown.", "-", "PRESSURE_RADIUS_M"),
    ("{t}_nearest_speed", "float32", "m/s", "Speed of the nearest player (local fit slope).",
     "Velocity never crosses a player jump.", "PLAYER_FIT_HALF"),
    ("{t}_nearest_relvel", "float32", "m/s", "|ball velocity - nearest player velocity|. Small "
     "means the ball moves with that player (dribble / carry).", "-",
     "BALL_FIT_HALF, PLAYER_FIT_HALF"),
    ("{t}_centroid_x", "float32", "m", "Mean x of the tracked players of this team.",
     "Cleaned positions; includes off-camera slots.", "-"),
    ("{t}_centroid_y", "float32", "m", "Mean y of the tracked players of this team.",
     "As {t}_centroid_x.", "-"),
    ("{t}_min_x", "float32", "m", "Smallest x among the tracked players of this team.",
     "As {t}_centroid_x.", "-"),
    ("{t}_max_x", "float32", "m", "Largest x among the tracked players of this team.",
     "As {t}_centroid_x.", "-"),
    ("{t}_mean_speed", "float32", "m/s", "Mean speed of the tracked players of this team that "
     "have a velocity. Low for both teams = likely stoppage.", "-", "PLAYER_FIT_HALF"),
    # Overall
    ("nearest_team", "category", "-", "'H' or 'A': team of the nearest tracked player to the "
     "smoothed ball; 'none' when the ball or both teams are unknown.", "-", "-"),
    ("nearest_dist", "float32", "m", "min(h_nearest_dist, a_nearest_dist).", "-", "-"),
    ("nearest_gap", "float32", "m", "a_nearest_dist - h_nearest_dist. Positive = home closer. NaN "
     "when either is unknown.", "-", "-"),
    # Smart Tagging
    ("st_{code}", "bool", "-", "A Smart Tagging interval with this code covers the frame "
     "(start inclusive, end exclusive, starter/load_smart_tagging.py).", "As given (automatic "
     "tagging, has false positives and negatives).", "-"),
    ("st_{code}_side", "category", "-", "Raw Team tag of the covering interval(s): 'H' "
     "(Barcelona), 'A' (Athletic, spelled 'Atletic Club' in the file), 'NA' (tag says N/A), "
     "'none' (code not active), 'MIX' (overlapping intervals of this code disagree). Raw: NOT "
     "inverted. Confirmed meaning: on DEFENDING codes it is the defending side. Other codes: "
     "meaning still to confirm.", "-", "-"),
    ("st_n_active", "int8", "-", "Number of Smart Tagging intervals covering the frame.", "-",
     "-"),
    ("st_set_piece_type", "category", "-", "Type tag of the covering SET PIECES interval, as a "
     "slug (throw_in, corner_kick, goal_kick, free_kick, kick_off_after_goal, kick_off_start), "
     "'none', or 'MIX'.", "-", "-"),
]


def expand_columns(codes):
    """Concrete column names, in table order, for the given Smart Tagging codes."""
    names = []
    for name, *_ in COLUMNS:
        if "{t}" in name:
            names += [name.format(t=t) for t, _ in TEAMS]
        elif "{code}" in name:
            names += [name.format(code=slug(c)) for c in codes]
        else:
            names.append(name)
    return names


# ------------------------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------------------------

def _pieces(idx, x, y, vmax_frame, tol):
    """Piece id per kept sample: a new piece starts after an impossible jump."""
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    step = np.hypot(np.diff(x[idx]), np.diff(y[idx]))
    brk = step > vmax_frame * np.diff(idx) + tol
    return np.concatenate([[0], np.cumsum(brk)])


def _heals(idx, x, y, pos_a, pos_b, vmax_frame, tol):
    """Can the object get from sample pos_a to sample pos_b at legal speed?"""
    a, b = idx[pos_a], idx[pos_b]
    return np.hypot(x[b] - x[a], y[b] - y[a]) <= vmax_frame * (b - a) + tol


def remove_spikes(x, y, max_speed, tol, min_piece):
    """(keep, piece) for one track. keep: samples that survive; piece: id per frame, -1 if none.

    Pieces are split by impossible jumps. A short piece is dropped if the pieces around it line
    up across it (single short pieces first, then runs of short pieces between two long ones);
    a short piece at either end is dropped if its only neighbour is long. Repeated until stable.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    vmax = max_speed / FPS
    keep = np.isfinite(x) & np.isfinite(y)
    for _ in range(MAX_CLEAN_PASSES):
        idx = np.flatnonzero(keep)
        if len(idx) < 2:
            break
        piece = _pieces(idx, x, y, vmax, tol)
        n_p = int(piece[-1]) + 1
        if n_p < 2:
            break
        ids = np.arange(n_p)
        first = np.searchsorted(piece, ids, side="left")
        last = np.searchsorted(piece, ids, side="right") - 1
        length = last - first + 1
        short = length < min_piece
        drop = np.zeros(n_p, dtype=bool)

        inner = ids[1:-1]
        if len(inner):
            drop[inner] = short[inner] & _heals(idx, x, y, last[inner - 1], first[inner + 1],
                                                vmax, tol)
        if short[0] and length[1] >= min_piece:
            drop[0] = True
        if short[-1] and length[-2] >= min_piece:
            drop[-1] = True
        drop[1:] &= ~drop[:-1].copy()     # never two neighbours in one pass

        if not drop.any():
            edges = np.flatnonzero(np.diff(np.concatenate([[0], short.astype(np.int8), [0]])))
            starts, ends = edges[0::2], edges[1::2] - 1
            ok = (starts > 0) & (ends < n_p - 1)
            starts, ends = starts[ok], ends[ok]
            if len(starts):
                heal = _heals(idx, x, y, last[starts - 1], first[ends + 1], vmax, tol)
                for a, b in zip(starts[heal], ends[heal]):
                    drop[a:b + 1] = True
        if not drop.any():
            break
        keep[idx[drop[piece]]] = False

    piece_of_frame = np.full(len(x), -1, dtype=np.int64)
    idx = np.flatnonzero(keep)
    if len(idx):
        piece_of_frame[idx] = _pieces(idx, x, y, vmax, tol)
    return keep, piece_of_frame


def gap_lengths(missing):
    """Per frame: length of the run of True in `missing` containing it, 0 where False."""
    missing = np.asarray(missing, dtype=bool)
    out = np.zeros(len(missing), dtype=np.int32)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], missing.astype(np.int8), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        out[a:b] = b - a
    return out


def fill_short_gaps(x, y, piece, max_frames):
    """Linear fill of gaps of at most max_frames between two samples of the same piece.

    Returns (x, y, piece, filled). Filled frames inherit the piece id.
    """
    x, y, piece = x.copy(), y.copy(), piece.copy()
    filled = np.zeros(len(x), dtype=bool)
    idx = np.flatnonzero(piece >= 0)
    if len(idx) < 2:
        return x, y, piece, filled
    gap = np.diff(idx) - 1
    ok = (gap > 0) & (gap <= max_frames) & (piece[idx[:-1]] == piece[idx[1:]])
    mark = np.zeros(len(x) + 1, dtype=np.int32)
    np.add.at(mark, idx[:-1][ok] + 1, 1)
    np.add.at(mark, idx[1:][ok], -1)
    filled = np.cumsum(mark)[:len(x)] > 0
    f = np.flatnonzero(filled)
    x[f] = np.interp(f, idx, x[idx])
    y[f] = np.interp(f, idx, y[idx])
    piece[f] = piece[idx[np.searchsorted(idx, f) - 1]]
    return x, y, piece, filled


def local_fit(v, piece, half, min_points):
    """Centred straight-line fit over +-half frames, per frame (works on 1-D or 2-D, axis 0).

    Only samples of the same piece as the centre are used, and the centre must be valid.
    Returns (level, slope_per_frame); NaN where fewer than min_points samples.
    """
    v = np.asarray(v, dtype=np.float64)
    valid = (piece >= 0) & np.isfinite(v)
    vv = np.where(valid, v, 0.0)
    s0, s1, s2, sv, stv = (np.zeros(v.shape) for _ in range(5))
    n = v.shape[0]
    for k in range(-half, half + 1):
        lo, hi = max(0, -k), min(n, n - k)
        if lo >= hi:
            continue
        i, j = slice(lo, hi), slice(lo + k, hi + k)
        m = (valid[j] & valid[i] & (piece[j] == piece[i])).astype(np.float64)
        s0[i] += m
        s1[i] += m * k
        s2[i] += m * k * k
        sv[i] += m * vv[j]
        stv[i] += m * k * vv[j]
    den = s0 * s2 - s1 * s1
    ok = valid & (s0 >= min_points) & (den > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(ok, (s0 * stv - s1 * sv) / den, np.nan)
        level = np.where(ok, (sv - slope * s1) / s0, np.nan)
    return level, slope


def clean_ball(ball_norm):
    """Dict of ball columns (numpy arrays) from the raw ATD ball (n, >=2), normalised 0-1."""
    bx = ball_norm[:, 0].astype(np.float64) * PITCH_X
    by = ball_norm[:, 1].astype(np.float64) * PITCH_Y
    missing = ~(np.isfinite(bx) & np.isfinite(by))
    keep, piece = remove_spikes(bx, by, BALL_MAX_SPEED_MS, BALL_POS_TOL_M, MIN_PIECE_FRAMES)
    jump = ~missing & ~keep
    gap_len = gap_lengths(~keep)
    xc, yc = np.where(keep, bx, np.nan), np.where(keep, by, np.nan)
    xc, yc, piece, interp = fill_short_gaps(xc, yc, piece, BALL_INTERP_MAX_FRAMES)

    lx, sx = local_fit(xc, piece, BALL_FIT_HALF, BALL_FIT_HALF + 1)
    ly, sy = local_fit(yc, piece, BALL_FIT_HALF, BALL_FIT_HALF + 1)
    xs = np.where(np.isfinite(lx), lx, xc)
    ys = np.where(np.isfinite(ly), ly, yc)
    vx, vy = sx * FPS, sy * FPS
    speed = np.hypot(vx, vy)

    pv = np.where(np.isfinite(vx) & np.isfinite(vy), piece, -1)
    _, ax = local_fit(vx, pv, BALL_FIT_HALF, BALL_FIT_HALF + 1)
    _, ay = local_fit(vy, pv, BALL_FIT_HALF, BALL_FIT_HALF + 1)
    accel = np.hypot(ax, ay) * FPS

    n, L = len(bx), BALL_TURN_LAG
    turn = np.full(n, np.nan)
    if n > 2 * L:
        c = np.arange(L, n - L)
        a, b = c - L, c + L
        ok = ((pv[a] >= 0) & (pv[a] == pv[b]) & (speed[a] >= BALL_TURN_MIN_SPEED_MS)
              & (speed[b] >= BALL_TURN_MIN_SPEED_MS))
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = (vx[a] * vx[b] + vy[a] * vy[b]) / (speed[a] * speed[b])
        turn[c] = np.where(ok, np.arccos(np.clip(cos, -1.0, 1.0)), np.nan)

    m = BALL_OUT_MARGIN_M
    with np.errstate(invalid="ignore"):
        out_x = (xc < -m) | (xc > PITCH_X + m)
        out_y = (yc < -m) | (yc > PITCH_Y + m)
    edge = np.minimum.reduce([xc, PITCH_X - xc, yc, PITCH_Y - yc])

    return {
        "ball_x": bx, "ball_y": by, "ball_missing": missing, "ball_jump": jump,
        "ball_interp": interp, "ball_gap_len": gap_len, "ball_x_c": xc, "ball_y_c": yc,
        "ball_x_s": xs, "ball_y_s": ys, "ball_vx": vx, "ball_vy": vy, "ball_speed": speed,
        "ball_accel": accel, "ball_turn": turn, "ball_out_x": out_x, "ball_out_y": out_y,
        "ball_out": out_x | out_y, "ball_edge_dist": edge,
    }


def clean_players(players_norm):
    """Cleaned players. players_norm (n, slots, >=2), normalised 0-1.

    Returns dict of (n, slots) arrays: x, y (cleaned, NaN if absent or dropped), px, py (fit
    level, else cleaned), vx, vy, speed (m/s), dropped (bool).
    """
    x = players_norm[:, :, 0].astype(np.float64) * PITCH_X
    y = players_norm[:, :, 1].astype(np.float64) * PITCH_Y
    present = np.isfinite(x) & np.isfinite(y)
    n, n_slots = x.shape
    out = {k: np.full((n, n_slots), np.nan, dtype=np.float32)
           for k in ("x", "y", "px", "py", "vx", "vy")}
    dropped = np.zeros((n, n_slots), dtype=bool)
    for s in range(n_slots):
        if not present[:, s].any():
            continue
        keep, piece = remove_spikes(x[:, s], y[:, s], PLAYER_MAX_SPEED_MS, PLAYER_POS_TOL_M,
                                    MIN_PIECE_FRAMES)
        dropped[:, s] = present[:, s] & ~keep
        xc, yc = np.where(keep, x[:, s], np.nan), np.where(keep, y[:, s], np.nan)
        lx, sx = local_fit(xc, piece, PLAYER_FIT_HALF, PLAYER_FIT_HALF + 1)
        ly, sy = local_fit(yc, piece, PLAYER_FIT_HALF, PLAYER_FIT_HALF + 1)
        out["x"][:, s], out["y"][:, s] = xc, yc
        out["px"][:, s] = np.where(np.isfinite(lx), lx, xc)
        out["py"][:, s] = np.where(np.isfinite(ly), ly, yc)
        out["vx"][:, s], out["vy"][:, s] = sx * FPS, sy * FPS
    out["speed"] = np.hypot(out["vx"], out["vy"])
    out["dropped"] = dropped
    return out


def in_view(x_m, y_m, matrices, hom_valid, image_size=IMAGE_SIZE):
    """(n, slots) bool: cleaned position projects inside the video image (homography valid)."""
    xn, yn = x_m / PITCH_X, y_m / PITCH_Y
    H = matrices
    h = lambda r, c: H[:, r, c][:, None]   # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        den = h(2, 0) * xn + h(2, 1) * yn + h(2, 2)
        u = (h(0, 0) * xn + h(0, 1) * yn + h(0, 2)) / den
        v = (h(1, 0) * xn + h(1, 1) * yn + h(1, 2)) / den
        inside = (np.isfinite(u) & np.isfinite(v) & (den > 0) & (u >= 0) & (u < image_size[0])
                  & (v >= 0) & (v < image_size[1]))
    return inside & np.asarray(hom_valid, dtype=bool)[:, None]


# ------------------------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------------------------

def _nanmean(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=1)


def _nanreduce(fn, a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return fn(a, axis=1)


def team_features(ball, pl, view, team_of_slot):
    """Per-team columns (dict) from cleaned ball dict, cleaned players dict and in-view mask."""
    out = {}
    bx, by = ball["ball_x_s"], ball["ball_y_s"]
    ball_ok = np.isfinite(bx) & np.isfinite(by)
    n = len(bx)
    rows = np.arange(n)
    for t, team in TEAMS:
        cols = np.flatnonzero(np.asarray(team_of_slot) == team)
        tracked = np.isfinite(pl["x"][:, cols])
        out[t + "_n_tracked"] = tracked.sum(axis=1).astype(np.int8)
        out[t + "_n_offcam"] = (tracked & ~view[:, cols]).sum(axis=1).astype(np.int8)
        out[t + "_n_dropped"] = pl["dropped"][:, cols].sum(axis=1).astype(np.int8)

        px = pl["px"][:, cols].astype(np.float64)
        py = pl["py"][:, cols].astype(np.float64)
        d = np.hypot(px - bx[:, None], py - by[:, None])
        dinf = np.where(np.isfinite(d), d, np.inf)
        if len(cols) == 0:
            dinf = np.full((n, 1), np.inf)
        order = np.argsort(dinf, axis=1, kind="stable")
        k1 = order[:, 0]
        d1 = dinf[rows, k1]
        d2 = dinf[rows, order[:, 1]] if dinf.shape[1] > 1 else np.full(n, np.inf)
        has = np.isfinite(d1)
        slot = cols[k1] if len(cols) else np.zeros(n, dtype=int)
        out[t + "_nearest_dist"] = np.where(has, d1, np.nan)
        out[t + "_nearest_slot"] = pd.array(np.where(has, slot + 1, 0), dtype="Int16")
        out[t + "_nearest_slot"][~has] = pd.NA
        offcam = pd.array(~view[rows, slot] if len(cols) else np.zeros(n, bool), dtype="boolean")
        offcam[~has] = pd.NA
        out[t + "_nearest_offcam"] = offcam
        out[t + "_second_dist"] = np.where(np.isfinite(d2), d2, np.nan)
        for r, name in ((CONTROL_RADIUS_M, "_n_within_2m"), (PRESSURE_RADIUS_M, "_n_within_5m")):
            cnt = pd.array((d <= r).sum(axis=1).astype(np.int8), dtype="Int8")
            cnt[~ball_ok] = pd.NA
            out[t + name] = cnt
        if len(cols):
            pvx, pvy = pl["vx"][rows, slot], pl["vy"][rows, slot]
            spd = pl["speed"][rows, slot]
        else:
            pvx = pvy = spd = np.full(n, np.nan)
        out[t + "_nearest_speed"] = np.where(has, spd, np.nan)
        out[t + "_nearest_relvel"] = np.where(
            has, np.hypot(ball["ball_vx"] - pvx, ball["ball_vy"] - pvy), np.nan)
        xs, ys = pl["x"][:, cols], pl["y"][:, cols]
        out[t + "_centroid_x"] = _nanmean(xs)
        out[t + "_centroid_y"] = _nanmean(ys)
        out[t + "_min_x"] = _nanreduce(np.nanmin, xs) if len(cols) else np.full(n, np.nan)
        out[t + "_max_x"] = _nanreduce(np.nanmax, xs) if len(cols) else np.full(n, np.nan)
        out[t + "_mean_speed"] = _nanmean(pl["speed"][:, cols])

    h, a = out["h_nearest_dist"], out["a_nearest_dist"]
    team = np.full(n, "none", dtype=object)
    team[np.isfinite(h) & ~(a <= h)] = "H"
    team[np.isfinite(a) & ~(h <= a)] = "A"
    team[np.isfinite(h) & np.isfinite(a) & (h == a)] = "H"     # exact tie: arbitrary, documented
    out["nearest_team"] = pd.Categorical(team, categories=["H", "A", "none"])
    out["nearest_dist"] = np.fmin(h, a)
    out["nearest_gap"] = a - h
    return out


def attack_direction(half, h_min_x, h_max_x, a_min_x, a_max_x):
    """Nullable boolean per frame: home attacks towards larger x in this half (NA outside)."""
    out = pd.array(np.zeros(len(half), dtype=bool), dtype="boolean")
    out[:] = pd.NA
    for hv in (1, 2):
        m = half == hv
        if not m.any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            s = ((np.nanmedian(a_min_x[m]) - np.nanmedian(h_min_x[m]))
                 + (np.nanmedian(a_max_x[m]) - np.nanmedian(h_max_x[m])))
        if np.isfinite(s) and s != 0:
            out[m] = bool(s > 0)
    return out


def st_side_code(team_tag, home_name):
    """Smart Tagging Team tag -> side code (home matched by substring, any other named club is
    away)."""
    t = (team_tag or "").strip().lower()
    if t in ("", "n/a", "na", "none"):
        return SIDE_NA
    h = (home_name or "").strip().lower()
    return SIDE_H if h and (t in h or h in t) else SIDE_A


def _tag(event, name):
    for key, value in event["tags"].items():
        if (key or "").strip().lower() == name.lower():
            return (value or "").strip()
    return None


def smart_tagging_features(events, n, home_name):
    """(columns dict, list of code names in column order)."""
    codes = list(KNOWN_ST_CODES)
    for e in events:
        c = (e["code"] or "").strip().upper()
        if c and c not in codes:
            codes.append(c)
    active = {c: np.zeros(n, dtype=bool) for c in codes}
    side = {c: np.zeros(n, dtype=np.int8) for c in codes}
    n_active = np.zeros(n, dtype=np.int16)
    sp_names = ["none"] + [slug(t) for t in KNOWN_SET_PIECE_TYPES]
    sp = np.zeros(n, dtype=np.int16)       # index into sp_names, -1 = MIX
    for e in events:
        c = (e["code"] or "").strip().upper()
        a, b = max(int(e["start_frame"]), 0), min(int(e["end_frame"]), n)
        if not c or a >= b:
            continue
        s = st_side_code(_tag(e, "Team"), home_name)
        active[c][a:b] = True
        cur = side[c][a:b]
        side[c][a:b] = np.where(cur == SIDE_NONE, s, np.where(cur == s, s, SIDE_MIX))
        n_active[a:b] += 1
        if c == "SET PIECES":
            t = slug(_tag(e, "Type") or "unknown")
            if t not in sp_names:
                sp_names.append(t)
            k = sp_names.index(t)
            cur = sp[a:b]
            sp[a:b] = np.where(cur == 0, k, np.where(cur == k, k, -1))
    out = {}
    for c in codes:
        out["st_" + slug(c)] = active[c]
    for c in codes:
        out["st_" + slug(c) + "_side"] = pd.Categorical(SIDE_NAMES[side[c]],
                                                        categories=list(SIDE_NAMES))
    out["st_n_active"] = np.minimum(n_active, 127).astype(np.int8)
    names = np.array(sp_names + ["MIX"], dtype=object)
    out["st_set_piece_type"] = pd.Categorical(names[np.where(sp < 0, len(sp_names), sp)],
                                              categories=list(names))
    return out, codes


def compute_features(ball_norm, players_norm, team_of_slot, half, matrices, hom_valid,
                     hom_error, events, home_name):
    """The full per-frame DataFrame. Pure: arrays in, DataFrame out, no file access."""
    n = len(ball_norm)
    half = np.asarray(half, dtype=np.int8)
    ball = clean_ball(ball_norm)
    pl = clean_players(players_norm)
    view = in_view(pl["x"], pl["y"], matrices, hom_valid)
    teams = team_features(ball, pl, view, team_of_slot)
    direction = attack_direction(half, teams["h_min_x"], teams["h_max_x"], teams["a_min_x"],
                                 teams["a_max_x"])
    d = direction.to_numpy(dtype=float, na_value=np.nan)
    x_att = np.where(d == 1, ball["ball_x_s"], np.where(d == 0, PITCH_X - ball["ball_x_s"],
                                                       np.nan))
    st, codes = smart_tagging_features(events, n, home_name)

    cols = {"frame": np.arange(n, dtype=np.int32), "half": half, "graded": half > 0,
            "hom_valid": np.asarray(hom_valid, dtype=bool),
            "hom_error": np.asarray(hom_error, dtype=np.float32)}
    cols.update(ball)
    cols["h_attacks_pos_x"] = direction
    cols["ball_x_att_h"] = x_att
    cols.update(teams)
    cols.update(st)

    df = pd.DataFrame({name: cols[name] for name in expand_columns(codes)})
    for name in df.columns:
        if df[name].dtype == np.float64:
            df[name] = df[name].astype(np.float32)
    return df


# ------------------------------------------------------------------------------------------------
# Loading (the starter loaders, plus a fast reader of the raw ATD text)
# ------------------------------------------------------------------------------------------------

def half_array(n):
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in half_video_blocks(GAME):
        half[first:last + 1] = h
    return half


def load_atd(raw_path, n_frames, n_slots):
    """(ball, players): ball (n, 3) and players (n, slots, 3), NaN where missing.

    Row i is video frame i (ATD frame_id is 1-based, so row = frame_id - 1).
    """
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    with open(raw_path) as f:
        for line in f:
            frame_id, player_part, ball_part = line.rstrip("\n").split(":")
            row = int(frame_id) - 1
            ball[row] = np.array(ball_part.split(","), dtype=np.float32)
            values = np.array(player_part.replace(";", ",").split(","), dtype=np.float32)
            players[row] = values.reshape(n_slots, 3)
    return ball, players


def load_inputs():
    import load_homography
    import load_smart_tagging
    from baseline import team_names, team_of_each_track

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n = source_frames()
    team_of_slot = np.asarray(team_of_each_track(meta_path))
    home_name, _ = team_names(meta_path)
    ball, players = load_atd(raw_path, n, len(team_of_slot))
    matrices, error, valid = load_homography.load()
    events = load_smart_tagging.load()
    return {"ball": ball, "players": players, "team_of_slot": team_of_slot, "half": half_array(n),
            "matrices": matrices, "hom_valid": valid, "hom_error": error, "events": events,
            "home_name": home_name, "meta_path": meta_path}


# ------------------------------------------------------------------------------------------------
# Data dictionary (written from COLUMNS and CONSTANTS; reads no data)
# ------------------------------------------------------------------------------------------------

def dictionary_markdown():
    md = ["# Feature table: data dictionary",
          "",
          "Generated by `python -m src.features --dictionary-only` from the column spec in that file. "
          "Do not edit by hand; edit `COLUMNS` / `CONSTANTS` and regenerate. Contains no data "
          "values.",
          "",
          "Table: `reports/features.parquet` (gitignored, local only). One row per tactical video "
          "frame, `frame` 0 .. 199318, same index as the submission.",
          "",
          "Build: `python -m src.features`",
          "",
          "## Coordinates",
          "",
          "- Metres. `x` = ATD normalised x * {} m (pitch length, `games.py`), `y` = ATD "
          "normalised y * {} m (pitch width).".format(PITCH_X, PITCH_Y),
          "- Origin = the pitch corner at ATD (0, 0). `x` runs along the touchlines from one goal "
          "line (x = 0) to the other (x = {}); `y` runs along the goal lines from one touchline "
          "(y = 0) to the other (y = {}).".format(PITCH_X, PITCH_Y),
          "- Attacking direction is not fixed: use `h_attacks_pos_x` (derived per half) or "
          "`ball_x_att_h`.",
          "- Times: 25 frames per second. Velocities m/s, accelerations m/s^2, angles radians.",
          "",
          "## Sources and what is not used",
          "",
          "- ATD raw text (players and ball, `src/features.load_atd`), ATD XML (slot "
          "teams, `starter/baseline.team_of_each_track`), homography (`starter/load_homography`), "
          "Smart Tagging (`starter/load_smart_tagging`).",
          "- Not used: the ATD `speed_kph` columns (they match neither km/h nor m/s), the public "
          "GT (never read, not for features, not for thresholds), the broadcast recording.",
          "",
          "## Columns",
          "",
          "`{t}` = `h` (home, Barcelona) or `a` (away, Athletic): one column per team. "
          "`{code}` = slug of a Smart Tagging code (list below).",
          "",
          "| column | type | unit | definition | cleaning | constants |",
          "|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "\\|")   # noqa: E731
    for row in COLUMNS:
        md.append("| `{}` | {} |".format(esc(row[0]), " | ".join(esc(c) for c in row[1:])))
    md += ["",
           "Smart Tagging codes (`{code}` slugs): " + ", ".join(
               "`{}` ({})".format(slug(c), c) for c in KNOWN_ST_CODES) + ". A code not in this "
           "list that appears in the file gets its own pair of columns at the end of the "
           "Smart Tagging block.",
           "",
           "Set-piece types: " + ", ".join("`{}` ({})".format(slug(t), t)
                                           for t in KNOWN_SET_PIECE_TYPES) + ".",
           "",
           "## Constants",
           "",
           "| name | value | unit | reasoning |",
           "|---|---|---|---|"]
    for name, value, unit, why in CONSTANTS:
        md.append("| `{}` | {} | {} | {} |".format(name, value, unit, esc(why)))
    md += ["",
           "## Cleaning notes",
           "",
           "- **Jumps** (ball and every player slot): consecutive observations further apart than "
           "max speed x elapsed time + tolerance split a track into pieces. Short pieces whose "
           "neighbours line up across them are dropped; a short piece at either end is dropped "
           "when its only neighbour is long. Long wrong-object pieces cannot be told apart from "
           "real movement and are kept; velocities never cross a piece boundary.",
           "- **Ball gaps**: only gaps up to `BALL_INTERP_MAX_FRAMES` inside one piece are filled "
           "(`ball_interp`). Players are not interpolated.",
           "- **Smoothing / velocity**: centred straight-line fit (local linear regression) over "
           "the cleaned points of the same piece; level = smoothed position, slope = velocity. "
           "Acceleration = the same fit applied to the velocity.",
           "- **Off-camera players** are kept (the tracker extrapolates them) but counted in "
           "`{t}_n_offcam` and flagged on the nearest player (`{t}_nearest_offcam`).",
           "",
           "## Known defects",
           "",
           "- The ATD ball has no height: aerial balls are projected to the ground and can look "
           "fast or jumpy. Never trust a single frame.",
           "- The ATD ball position is noisy, sometimes behind or beside the real ball.",
           "- Off-camera players are unreliable; a keeper slot can sit on the wrong player.",
           "- Slots are tracklets, not people: `{t}_nearest_slot` can change while the same "
           "player keeps the ball.",
           "- Smart Tagging is automatic (false positives and negatives). Team tag meaning is "
           "confirmed only for DEFENDING codes (defending side). Throw-ins and kick-offs are "
           "always N/A.",
           ""]
    return "\n".join(md)


def write_dictionary(path=DEFAULT_DICTIONARY):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(dictionary_markdown())


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------

def resolve_report_path(path):
    """Absolute path inside reports/ (the table is match data)."""
    out = os.path.realpath(path if os.path.isabs(path) else os.path.join(ROOT, path))
    reports = os.path.realpath(REPORT_DIR)
    if os.path.commonpath([out, reports]) != reports:
        raise ValueError("output must be inside reports/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the per-frame feature table.")
    p.add_argument("--out", default=DEFAULT_TABLE, help="parquet path, inside reports/")
    p.add_argument("--dictionary", default=DEFAULT_DICTIONARY, help="data dictionary path")
    p.add_argument("--dictionary-only", action="store_true",
                   help="only write the data dictionary (reads no data)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    write_dictionary(args.dictionary)
    if args.dictionary_only:
        print("dictionary written")
        return
    out_path = resolve_report_path(args.out)
    inputs = load_inputs()
    df = compute_features(inputs["ball"], inputs["players"], inputs["team_of_slot"],
                          inputs["half"], inputs["matrices"], inputs["hom_valid"],
                          inputs["hom_error"], inputs["events"], inputs["home_name"])
    df.to_parquet(out_path, index=False)
    print("features written")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        # Type and location only: an exception message can quote data.
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        print("FAILED: {} at {}:{}".format(type(exc).__name__, os.path.basename(frame.filename),
                                           frame.lineno))
        sys.exit(1)
