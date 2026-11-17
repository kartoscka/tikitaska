"""Training data for the learnt vote: Metrica Sports' public sample games 1 and 2.

Why
    The challenge gives one match, with right answers for only 5 minutes, and those 5 minutes may
    not be used for training or tuning. The rules allow data about other games, so the learnt vote of
    rule_learnt_ensemble is fitted on two other full matches.

Source
    github.com/metrica-sports/sample-data, pinned to the commit below. Published by Metrica Sports
    for the football analytics community: anonymised matches, tracking at 25 fps plus an event log
    (every pass, shot, recovery, ball out, set piece...). Their readme asks to credit the source when
    it is used publicly: data by Metrica Sports. The files are downloaded on first use into
    data/external/metrica/ (git-ignored, like all match data).

Labels
    The samples have no possession labels. They are built from the event log with a fixed rule
    (build_labels, below) and kept in external/metrica_samples/ (one row per spell of H, A or D).
    The rule follows the challenge's definition: possession changes only when the other team gets
    control. These labels come from events, the organisers' from corrected tracking: a close
    stand-in, not the same thing.

Made to look like BAR-ATH
    The samples' ball is cleaner than BAR-ATH's and is missing mostly when play is stopped. Before
    training, the ball is shifted the way BAR-ATH's tracking puts it (6 frames ahead, a slow drift of
    0.65 m), and stretches of ball are deleted during live play with BAR-ATH's own gap lengths:
    0%, 15% and 30% of the live ball. The restart tags (from the event log) are thinned to as many
    set-piece tags per minute of play as BAR-ATH's Smart Tagging has, and moved by up to 2 s
    (sparse_restarts). BAR-ATH's feature table is read only for those two things: gap lengths and
    tags per minute. No BAR-ATH label is used.

Use
    python -m src.training.metrica_samples                   # download the sample files
    python -m src.training.metrica_samples --write-labels    # rebuild the label files
"""

import argparse
import copy
import os
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src import features as F  # noqa: E402   the same features as BAR-ATH

SOURCE = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
          "e706dd506b360d69d9d123d5b8026e7294b13996/data/Sample_Game_{g}/")
FILES = ("Sample_Game_{g}_RawTrackingData_Home_Team.csv", "Sample_Game_{g}_RawTrackingData_Away_Team.csv",
         "Sample_Game_{g}_RawEventsData.csv")
GAMES = (1, 2)
KEEP_RATE_GAMES = (1, 2)       # the games the restart keep rate is computed on
CACHE = os.path.join(ROOT, "data", "external", "metrica")
LABELS = os.path.join(ROOT, "external", "metrica_samples")

FPS = 25
HOME_NAME, AWAY_NAME = "Home", "Away"
LEAD_FRAMES = 6                # BAR-ATH's ball sits about 6 frames ahead of the samples' ...
DRIFT_M = 0.65                 # ... and drifts slowly by about 0.65 m
CONTROL_MIN_S = 2.0            # labels: a spell this long counts as control even without a pass
CONTROL_TYPES = ("PASS", "SHOT", "SET PIECE")
RESTART_JITTER_S = 2.0         # sparse_restarts: each kept set-piece tag moves by up to 2 s


# --- the files --------------------------------------------------------------------------------


def _files(game):
    return [name.format(g=game) for name in FILES]


def fetch():
    """Download the sample files that are not in data/external/metrica/ yet (games 1 and 2)."""
    os.makedirs(CACHE, exist_ok=True)
    for g in GAMES:
        for name in _files(g):
            path = os.path.join(CACHE, name)
            if os.path.exists(path):
                continue
            print("downloading", name)
            tmp = path + ".part"
            urllib.request.urlretrieve(SOURCE.format(g=g) + name, tmp)
            os.replace(tmp, path)


def _tracking(path):
    """(period, positions (n, k, 2) in 0-1 pitch units, ball (n, 2)) from one team's CSV."""
    raw = pd.read_csv(path, skiprows=2)
    period = raw["Period"].to_numpy(np.int8)
    cols = list(raw.columns)
    b = cols.index("Ball")
    players = [c for c in cols[3:b] if c.startswith("Player")]
    pos = np.stack([raw[[c, cols[cols.index(c) + 1]]].to_numpy(np.float32) for c in players], axis=1)
    ball = raw[[cols[b], cols[b + 1]]].to_numpy(np.float32)
    return period, pos, ball


def _tracks(game):
    """(period [n], players [n, slots, 2] in 0-1 pitch units, team_of_slot, ball [n, 2], player
    numbers as text), home slots first."""
    period, home, ball = _tracking(os.path.join(CACHE, FILES[0].format(g=game)))
    _, away, _ = _tracking(os.path.join(CACHE, FILES[1].format(g=game)))
    numbers = []
    for name in FILES[:2]:
        cols = pd.read_csv(os.path.join(CACHE, name.format(g=game)), skiprows=2, nrows=0).columns
        numbers += [c.replace("Player", "") for c in cols if c.startswith("Player")]
    team_of_slot = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    return period, np.concatenate([home, away], axis=1), team_of_slot, ball, numbers


def _events(game):
    ev = pd.read_csv(os.path.join(CACHE, FILES[2].format(g=game)))
    ev["Subtype"] = ev["Subtype"].fillna("").astype(str).str.upper()
    ev["Type"] = ev["Type"].astype(str).str.upper()
    return ev.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)


def _restart_event(code, a, b, team, kind):
    """One tag in the same shape as a Smart Tagging tag (what src/features.py reads)."""
    return {"code": code, "start_frame": int(a), "end_frame": int(b), "start_s": a / FPS,
            "end_s": b / FPS, "tags": {"Team": team or "N/A", "Type": kind}}


# --- the labels, from the event log -----------------------------------------------------------


def _walk_events(ev, period, n):
    """Restart tags for the models, and the first answers, walking through the event log in time order:
      - a PASS, SHOT, RECOVERY, SET PIECE or won duel or dribble gives the ball to its team;
      - BALL LOST alone changes nothing: a loose ball stays with the last team until the other team
        records an event;
      - BALL OUT, a foul (FAULT RECEIVED), a shot that ends out or in a goal, and the end of a half
        make the ball dead until the next restart (a SET PIECE, or a goal-kick PASS)."""
    labels = np.full(n, "D", dtype="<U1")
    tags = []
    for half in sorted(set(period.tolist()) - {0}):
        idx = np.flatnonzero(period == half)
        a0, b0 = int(idx[0]), int(idx[-1]) + 1
        state, dead, cursor = "D", True, a0            # dead until the kick-off
        for _, e in ev[ev["Period"] == half].iterrows():
            f = int(e["Start Frame"]) - 1
            g = int(e["End Frame"]) - 1 if e["End Frame"] > 0 else f
            f, g = min(max(f, a0), b0), min(max(g, a0), b0)
            labels[cursor:f] = state
            cursor = f
            team = "H" if e["Team"] == HOME_NAME else "A"
            typ, sub = e["Type"], e["Subtype"]
            if typ == "SET PIECE" or (typ == "PASS" and "GOAL KICK" in sub):
                kind = "Goal Kick" if typ == "PASS" else sub.title()
                tags.append(_restart_event("SET PIECES", f, max(g, f) + 3 * FPS,
                                           HOME_NAME if team == "H" else AWAY_NAME, kind))
                state, dead = team, False
                continue
            if dead:
                continue                              # nothing changes possession while dead
            if typ in ("PASS", "SHOT", "RECOVERY"):
                state = team
            elif typ == "CHALLENGE" and ("GROUND-WON" in sub or "DRIBBLE-WON" in sub):
                state = team
            if typ == "BALL OUT" or typ == "FAULT RECEIVED" or "END HALF" in sub:
                stop = g if typ == "BALL OUT" else f
                labels[cursor:stop] = state
                cursor, state, dead = stop, "D", True
            elif typ == "SHOT" and ("OUT" in sub or "GOAL" in sub):
                labels[cursor:g] = state
                cursor, state, dead = g, "D", True
                if "GOAL" in sub:
                    tags.append(_restart_event("GOALS", f, g + FPS, HOME_NAME if team == "H" else AWAY_NAME, ""))
        labels[cursor:b0] = state
    return sorted(tags, key=lambda t: t["start_frame"]), labels


def _controlled(labels, ev):
    """The challenge's definition: possession changes only when the other team gets control. A spell
    that follows the other team's possession (no dead ball between) counts only if its team plays
    the ball on purpose in it (a PASS, SHOT or SET PIECE starts in it) or keeps it CONTROL_MIN_S or
    more. Otherwise it was a touch without control (a deflection, a duel won without keeping the
    ball) and the ball stays with the previous team. Repeated until nothing changes."""
    out = labels.copy()
    starts = {t: np.sort(ev.loc[(ev["Team"] == name) & ev["Type"].isin(CONTROL_TYPES), "Start Frame"]
                         .to_numpy(np.int64) - 1) for t, name in (("H", HOME_NAME), ("A", AWAY_NAME))}
    for _ in range(100):
        changed = False
        edges = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate([[0], edges, [len(out)]])
        for i in range(1, len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            t, prev = out[a], out[a - 1]
            if t == "D" or prev == "D" or t == prev:
                continue
            s = starts[t]
            if np.searchsorted(s, a) >= np.searchsorted(s, b) and (b - a) < CONTROL_MIN_S * FPS:
                out[a:b] = prev
                changed = True
        if not changed:
            break
    return out


def build_labels(game):
    """The right answers of one sample game, one letter per frame, rebuilt from its event log."""
    fetch()
    period = _tracks(game)[0]
    ev = _events(game)
    _, labels = _walk_events(ev, period, len(period))
    return _controlled(labels, ev)


def label_path(game):
    return os.path.join(LABELS, "Sample_Game_{}_labels.csv".format(game))


def write_labels():
    """Rebuild external/metrica_samples/Sample_Game_<g>_labels.csv: one row per spell of H, A or D
    (start_frame included, end_frame excluded, 0-based frames of the tracking files)."""
    os.makedirs(LABELS, exist_ok=True)
    for g in GAMES:
        lab = build_labels(g)
        edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        a = np.concatenate([[0], edges])
        b = np.concatenate([edges, [len(lab)]])
        pd.DataFrame({"start_frame": a, "end_frame": b, "label": lab[a]}).to_csv(label_path(g), index=False)
        print("wrote", os.path.relpath(label_path(g), ROOT))


def load_labels(game):
    """The committed right answers of one sample game, one letter per frame."""
    runs = pd.read_csv(label_path(game))
    return np.repeat(runs["label"].to_numpy(dtype="<U1"), (runs["end_frame"] - runs["start_frame"]).to_numpy())


# --- the inputs, as the model sees them -------------------------------------------------------


def load_inputs(game):
    """Tracking and restart tags of one sample game, in the shape src/features.py takes."""
    fetch()
    period, pos, team_of_slot, ball, _ = _tracks(game)
    n = len(period)
    players = np.full((n, pos.shape[1], 3), np.nan, dtype=np.float32)
    players[:, :, :2] = pos
    ball3 = np.full((n, 3), np.nan, dtype=np.float32)
    ball3[:, :2] = ball
    tags, _ = _walk_events(_events(game), period, n)
    return {"ball": ball3, "players": players, "team_of_slot": team_of_slot, "half": period,
            # the whole pitch is tracked: a camera that sees all of it keeps every player in view
            "matrices": np.tile(np.diag([1919.0, 1079.0, 1.0]), (n, 1, 1)),
            "hom_valid": np.ones(n, bool), "hom_error": np.zeros(n), "events": tags}


def _runs(mask):
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    e = np.flatnonzero(np.diff(m))
    return list(zip(e[0::2], e[1::2]))


def bar_ath_gap_lengths(bar_ath_features):
    """Lengths (frames) of BAR-ATH's ball gaps that the features do not bridge. Tracking only."""
    m = bar_ath_features["ball_missing"].to_numpy(dtype=bool) & (bar_ath_features["half"].to_numpy() > 0)
    return np.array([b - a for a, b in _runs(m) if b - a > F.BALL_INTERP_MAX_FRAMES])


def bar_ath_restart_keep(bar_ath_features):
    """Share of the samples' set-piece tags to keep so that they are as sparse as BAR-ATH's Smart
    Tagging: BAR-ATH's set pieces per minute of play over the samples' (at most 1)."""
    graded = bar_ath_features["half"].to_numpy() > 0
    sp = bar_ath_features["st_set_pieces"].to_numpy(dtype=bool) & graded
    bar = len(_runs(sp)) / (graded.sum() / FPS / 60)
    rates = []
    for g in KEEP_RATE_GAMES:
        inp = load_inputs(g)
        n_sp = sum(1 for e in inp["events"] if e["code"] == "SET PIECES")
        rates.append(n_sp / ((np.asarray(inp["half"]) > 0).sum() / FPS / 60))
    return float(min(1.0, bar / np.mean(rates)))


def sparse_restarts(events, keep, jitter_s, seed):
    """Restart tags made as incomplete as Smart Tagging's: each set piece kept with probability
    `keep`, and its start moved by up to +-jitter_s seconds. Goals are left alone. Seeded."""
    rng = np.random.default_rng(seed)
    out = []
    for e in events:
        if (e["code"] or "").strip().upper() == "SET PIECES":
            if rng.random() > keep:
                continue
            e = dict(e)
            e["start_frame"] = max(0, int(e["start_frame"] + rng.uniform(-jitter_s, jitter_s) * FPS))
            e["end_frame"] = max(int(e["end_frame"]), e["start_frame"] + FPS)
        out.append(e)
    return out


def with_smart_tagging(df, base_events, tags):
    """A copy of a feature table with every st_* column rebuilt from the game's own tags plus `tags`."""
    cols, _ = F.smart_tagging_features(list(base_events) + list(tags), len(df), HOME_NAME)
    out = df.copy()
    for name, values in cols.items():
        out[name] = values
    return out


def like_bar_ath(inputs, labels, loss, gap_lengths, game):
    """The sample's ball made to look like BAR-ATH's: shifted LEAD_FRAMES ahead, a slow drift of
    DRIFT_M, and `loss` of the live ball deleted in gaps of BAR-ATH's lengths. Seeded: always the
    same result."""
    out = copy.copy(inputs)
    ball = inputs["ball"].copy()
    ball[:-LEAD_FRAMES] = ball[LEAD_FRAMES:]
    ball[-LEAD_FRAMES:] = np.nan
    n = len(ball)
    rng = np.random.default_rng(game)
    a = np.exp(-1 / FPS)                                  # the drift remembers about one second
    e = rng.normal(0, DRIFT_M * np.sqrt(1 - a * a), (n, 2))
    drift = np.zeros((n, 2))
    for f in range(1, n):
        drift[f] = a * drift[f - 1] + e[f]
    ball[:, 0] += (drift[:, 0] / F.PITCH_X).astype(np.float32)
    ball[:, 1] += (drift[:, 1] / F.PITCH_Y).astype(np.float32)
    if loss > 0:
        rng = np.random.default_rng(100 * game + int(100 * loss))
        live = (labels != "D") & (np.asarray(inputs["half"]) > 0)
        idx = np.flatnonzero(live)
        gone = np.zeros(n, bool)
        while (gone & live).sum() < loss * live.sum():
            s = int(rng.choice(idx))
            gone[s:min(s + int(rng.choice(gap_lengths)), n)] = True
        ball[gone] = np.nan
    out["ball"] = ball
    return out


def features(inputs):
    """The feature table of a sample game, built exactly like BAR-ATH's (src/features.py), plus each
    team's width across the pitch (x_h_width, x_a_width) and the player movement columns (pd_*). On
    BAR-ATH the model computes them from the raw tracking itself."""
    from src.models.rule_learnt_ensemble import pd_columns, width_columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = F.compute_features(inputs["ball"], inputs["players"], inputs["team_of_slot"], inputs["half"],
                                inputs["matrices"], inputs["hom_valid"], inputs["hom_error"],
                                inputs["events"], HOME_NAME)
    for name, col in width_columns(inputs["players"], inputs["team_of_slot"]).items():
        df[name] = col
    for name, col in pd_columns(inputs["players"], inputs["team_of_slot"], inputs["half"], df).items():
        df[name] = col
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Metrica sample games: download, and rebuild the labels.")
    ap.add_argument("--write-labels", action="store_true", help="rebuild external/metrica_samples/*.csv")
    if ap.parse_args().write_labels:
        write_labels()
    else:
        fetch()
