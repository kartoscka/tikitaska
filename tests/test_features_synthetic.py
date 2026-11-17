"""Synthetic check of src/features.py. Made-up arrays only: reads no match data.

Prints one PASS / FAIL line per check.
Run:  python tests/test_features_synthetic.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import features as F  # noqa: E402

RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def make_inputs(tmpdir):
    rng = np.random.default_rng(0)
    n, S = 3000, 6
    f = np.arange(n)
    half = np.zeros(n, dtype=np.int8)
    half[100:1500] = 1
    half[1500:] = 2

    # Ball: back and forth along x, constant y, small noise.
    period = 25 * 8
    bx = 50 + 20 * np.sin(2 * np.pi * f / period)
    by = np.full(n, 34.0)
    vtrue = np.abs(20 * 2 * np.pi / period * F.FPS * np.cos(2 * np.pi * f / period))
    bx_n = bx + rng.normal(0, 0.1, n)
    by_n = by + rng.normal(0, 0.1, n)
    by_n[1000:1050] = 68 + 3.0                 # ball out beyond a touchline
    ball_m = np.stack([bx_n, by_n], axis=1)
    ball_m[500] = [bx_n[500] + 30, by_n[500]]  # one-frame spike
    ball_m[600:610] = np.nan                   # short gap
    ball_m[800:860] = np.nan                   # long gap
    ball = np.full((n, 3), np.nan)
    ball[:, 0] = ball_m[:, 0] / F.PITCH_X
    ball[:, 1] = ball_m[:, 1] / F.PITCH_Y

    pl = np.full((n, S, 2), np.nan)
    keeper_left = np.where(half == 2, 103.0, 2.0)
    keeper_right = np.where(half == 2, 2.0, 103.0)
    pl[:, 0] = np.stack([keeper_left, np.full(n, 34.0)], 1)          # home keeper
    pl[:, 1] = np.stack([bx, by + 0.5], 1)                          # home carrier
    pl[:, 1, 1][1000:1050] = by_n[1000:1050] + 0.5
    pl[:, 2] = np.stack([np.full(n, 40.0), np.full(n, 20.0)], 1)    # home static
    pl[700, 2, 0] += 20                                             # one-frame player spike
    pl[:, 3] = np.stack([keeper_right, np.full(n, 34.0)], 1)         # away keeper
    pl[:, 4] = np.stack([bx, by + 6.0], 1)                          # away marker
    pl[:, 5] = np.stack([np.full(n, 63.0), np.full(n, 74.8)], 1)    # away, off camera
    players = np.full((n, S, 3), np.nan)
    players[:, :, 0] = pl[:, :, 0] / F.PITCH_X
    players[:, :, 1] = pl[:, :, 1] / F.PITCH_Y

    team_of_slot = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    matrices = np.tile(np.diag([1920.0, 1080.0, 1.0]), (n, 1, 1))
    valid = np.ones(n, dtype=bool)
    error = np.zeros(n)
    events = [
        {"code": "DEFENDING IN MIDDLE THIRD", "start_frame": 200, "end_frame": 300,
         "tags": {"Team": "Atletic Club"}},
        {"code": "SET PIECES", "start_frame": 300, "end_frame": 350,
         "tags": {"Team": "N/A", "Type": "Throw-in"}},
        {"code": "BUILD UP", "start_frame": 400, "end_frame": 450, "tags": {"Team": "Barcelona"}},
    ]
    meta = os.path.join(tmpdir, "meta.xml")
    with open(meta, "w") as fh:
        fh.write("<Root><ProviderParameter><Name>attack_direction_first_half</Name>"
                 "<Value>test</Value></ProviderParameter></Root>")
    return {"ball": ball, "players": players, "team_of_slot": team_of_slot, "half": half,
            "matrices": matrices, "hom_valid": valid, "hom_error": error, "events": events,
            "home_name": "FC Barcelona", "meta_path": meta}, vtrue, bx


def main():
    tmpdir = tempfile.mkdtemp()
    inp, vtrue, bx = make_inputs(tmpdir)
    df = F.compute_features(inp["ball"], inp["players"], inp["team_of_slot"], inp["half"],
                            inp["matrices"], inp["hom_valid"], inp["hom_error"], inp["events"],
                            inp["home_name"])

    codes = [c[3:-5] for c in df.columns if c.startswith("st_") and c.endswith("_side")]
    check("columns match the spec", list(df.columns) == F.expand_columns(F.KNOWN_ST_CODES))
    check("known codes all present", len(codes) == len(F.KNOWN_ST_CODES))
    md = F.dictionary_markdown()
    check("dictionary lists every column template",
          all("`{}`".format(row[0]) in md for row in F.COLUMNS))
    check("no float64 left", not any(df[c].dtype == np.float64 for c in df.columns))

    jump = df["ball_jump"].to_numpy()
    check("ball spike dropped", jump[500])
    check("no other ball jumps", jump[100:1500].sum() == 1)
    interp = df["ball_interp"].to_numpy()
    xc = df["ball_x_c"].to_numpy(dtype=float)
    check("short gap interpolated", interp[600:610].all())
    check("interpolation close to truth", np.nanmax(np.abs(xc[600:610] - bx[600:610])) < 0.5)
    check("long gap left missing", np.isnan(xc[800:860]).all() and not interp[800:860].any())
    check("gap length recorded", (df["ball_gap_len"].to_numpy()[600:610] == 10).all())

    sp = df["ball_speed"].to_numpy(dtype=float)
    ok = np.arange(120, 480)
    check("ball speed matches truth", np.nanmedian(np.abs(sp[ok] - vtrue[ok])) < 0.5)
    check("ball speed defined on clean frames", np.isfinite(sp[ok]).mean() > 0.95)
    check("accel finite somewhere", np.isfinite(df["ball_accel"].to_numpy(dtype=float)[ok]).any())

    check("nearest team is home", (df["nearest_team"].astype(str).to_numpy()[ok] == "H").all())
    check("nearest slot is the carrier",
          (df["h_nearest_slot"].to_numpy(dtype=float, na_value=np.nan)[ok] == 2).all())
    check("one home player within 2 m",
          (df["h_n_within_2m"].to_numpy(dtype=float, na_value=np.nan)[ok] == 1).all())
    check("ball moves with carrier",
          np.nanmedian(df["h_nearest_relvel"].to_numpy(dtype=float)[ok]) < 0.5)
    check("away nearest about 6 m",
          abs(np.nanmedian(df["a_nearest_dist"].to_numpy(dtype=float)[ok]) - 6.0) < 0.5)
    check("gap positive when home closer", (df["nearest_gap"].to_numpy(dtype=float)[ok] > 0).all())

    out_y = df["ball_out_y"].to_numpy()
    check("ball out beyond touchline", out_y[1005:1045].all())
    check("ball in play not out", not df["ball_out"].to_numpy()[ok].any())
    check("edge distance negative when out",
          (df["ball_edge_dist"].to_numpy(dtype=float)[1005:1045] < 0).all())

    check("player spike dropped", df["h_n_dropped"].to_numpy()[700] == 1)
    check("off-camera player counted", (df["a_n_offcam"].to_numpy()[ok] == 1).all())
    check("home tracked count", (df["h_n_tracked"].to_numpy()[ok] == 3).all())

    d = df["h_attacks_pos_x"]
    check("direction half 1", bool(d.iloc[200]) is True)
    check("direction half 2", bool(d.iloc[2000]) is False)
    check("direction NA outside halves", pd.isna(d.iloc[50]))
    xa = df["ball_x_att_h"].to_numpy(dtype=float)
    check("ball_x_att_h flips in half 2",
          abs(xa[2000] - (F.PITCH_X - df["ball_x_s"].to_numpy(dtype=float)[2000])) < 1e-3)

    check("defending side raw (A)", df["st_defending_in_middle_third_side"].iloc[250] == "A")
    check("side none when inactive", df["st_defending_in_middle_third_side"].iloc[100] == "none")
    check("set piece side NA", df["st_set_pieces_side"].iloc[320] == "NA")
    check("set piece type", df["st_set_piece_type"].iloc[320] == "throw_in")
    check("build up side H", df["st_build_up_side"].iloc[420] == "H")
    check("end exclusive", not df["st_build_up"].iloc[450])
    check("n active", df["st_n_active"].iloc[250] == 1)

    # remove_spikes: two consecutive spikes far from each other, inside a straight track.
    x = np.arange(100, dtype=float) * 0.3
    y = np.zeros(100)
    x[50], x[51] = 80.0, 40.0
    keep, piece = F.remove_spikes(x, y, F.BALL_MAX_SPEED_MS, F.BALL_POS_TOL_M,
                                  F.MIN_PIECE_FRAMES)
    check("two-spike run dropped", not keep[50] and not keep[51] and keep.sum() == 98)
    check("track reunited after spikes", len(np.unique(piece[keep])) == 1)

    path = os.path.join(tmpdir, "features.parquet")
    df.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    check("parquet roundtrip", back.shape == df.shape
          and all(str(back[c].dtype) == str(df[c].dtype) for c in df.columns))

    n_fail = sum(1 for _, ok_ in RESULTS if not ok_)
    print("ALL PASS" if n_fail == 0 else "FAILURES: {}".format(n_fail))
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
