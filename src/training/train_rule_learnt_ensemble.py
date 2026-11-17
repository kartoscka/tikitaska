"""Training script of rule_learnt_ensemble: refits the learnt vote and rewrites the fitted numbers.

    python -m src.training.train_rule_learnt_ensemble

Training sample (public data only): Metrica sample games 1 and 2, each at ball loss 0, 15 and 30 %
(src/training/metrica_samples.py): the ball made like BAR-ATH's (ms.like_bar_ath, seed = game), the
feature table (ms.features), the restart tags thinned like BAR-ATH's Smart Tagging
(ms.sparse_restarts, seed 7 * game), no phase tags, then the player movement columns. Every 5th frame
that has a label is a training row; the label comes from external/metrica_samples/.

BAR-ATH's feature table (reports/features.parquet, built by python -m src.features) is read only
through ms.bar_ath_gap_lengths and ms.bar_ath_restart_keep: the lengths of its ball gaps and how many
restart tags it has. No BAR-ATH label is used. The fit (multinomial logistic regression,
L2 = 1e-4, full-batch L-BFGS from zeros) uses numpy only and has no random part: the same file comes
out every time. Output: the block between BEGIN FITTED and END FITTED in
src/models/rule_learnt_ensemble.py.
"""

import ctypes
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.models import rule_learnt_ensemble as V

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_FILE = os.path.join(ROOT, "src", "models", "rule_learnt_ensemble.py")
GAMES = (1, 2)
LOSSES = (0.0, 0.15, 0.30)
L2 = 1e-4
BEGIN = "# --- BEGIN FITTED (written by src/training/train_rule_learnt_ensemble.py; do not edit) ---"
END = "# --- END FITTED ---"
WORKERS = 3


def _low_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:
        pass


def _metrica():
    from src.training import metrica_samples as ms
    return ms


def _samples(args):
    """One run (game, loss): the rows of the training sample. gap_lengths, keep: from BAR-ATH's table."""
    game, loss, gap_lengths, keep = args
    _low_priority()
    ms = _metrica()
    labels = ms.load_labels(game)
    base = ms.load_inputs(game)
    inp = ms.like_bar_ath(base, labels, loss, gap_lengths, game)
    df = ms.features(inp)
    events = ms.sparse_restarts(base["events"], keep, ms.RESTART_JITTER_S, seed=7 * game)
    df = ms.with_smart_tagging(df, events, [])
    for name, col in V.pd_columns(inp["players"], inp["team_of_slot"], inp["half"], df).items():
        df[name] = col
    X, names = V.features(df)
    code = {"H": 0, "A": 1, "D": 2}
    y = np.array([code.get(str(v).strip().upper(), -1) for v in labels], dtype=np.int64)
    rows = np.flatnonzero(y >= 0)[::V.STEP]
    return X[rows], y[rows], names


def build_sample():
    import pandas as pd
    ms = _metrica()
    bar = pd.read_parquet(os.path.join(ROOT, "reports", "features.parquet"))
    gap_lengths = ms.bar_ath_gap_lengths(bar)
    keep = ms.bar_ath_restart_keep(bar)
    del bar
    todo = [(g, l, gap_lengths, keep) for g in GAMES for l in LOSSES]
    with ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_samples, todo))                  # in order: game 1 then 2, loss 0, 15, 30
    names = parts[0][2]
    assert all(p[2] == names for p in parts), "the runs give different input columns"
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), names


def fit(X, y, n_classes=3, l2=L2, iters=300, tol=1e-9, mem=10):
    """Multinomial logistic regression, full-batch L-BFGS from zeros. Returns the fitted numbers."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    m, p = X.shape
    K = int(n_classes)
    nan = np.isnan(X)
    with np.errstate(all="ignore"):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    flag_cols = np.flatnonzero(nan.any(axis=0))
    del nan
    Z = V.standardise(X, mean, std, flag_cols)
    q = Z.shape[1]
    Y = np.zeros((m, K))
    Y[np.arange(m), y] = 1.0

    def fg(theta):
        W = theta[:q * K].reshape(q, K)
        b = theta[q * K:]
        P = V.softmax(Z @ W + b)
        loss = -np.log(np.maximum(P[np.arange(m), y], 1e-300)).mean() + 0.5 * l2 * (W * W).sum()
        P -= Y
        P /= m
        gW = Z.T @ P + l2 * W
        gb = P.sum(axis=0)
        return loss, np.concatenate([gW.ravel(), gb])

    theta = np.zeros(q * K + K)
    f, g = fg(theta)
    S, Yh = [], []
    for it in range(iters):
        d = -g.copy()
        alphas = []
        for s, yv in zip(reversed(S), reversed(Yh)):
            a = (s @ d) / (yv @ s)
            alphas.append(a)
            d -= a * yv
        if S:
            d *= (S[-1] @ Yh[-1]) / (Yh[-1] @ Yh[-1])
        else:
            d *= 1.0 / max(np.linalg.norm(g), 1e-12)
        for (s, yv), a in zip(zip(S, Yh), reversed(alphas)):
            bta = (yv @ d) / (yv @ s)
            d += s * (a - bta)
        gd = g @ d
        if gd >= 0:
            d, gd = -g, -(g @ g)
            S, Yh = [], []
        t = 1.0
        for _ in range(40):
            f2, g2 = fg(theta + t * d)
            if f2 <= f + 1e-4 * t * gd:
                break
            t *= 0.5
        else:
            break
        s = t * d
        yv = g2 - g
        theta = theta + s
        rel = abs(f - f2) / max(abs(f), 1e-12)
        f, g = f2, g2
        if s @ yv > 1e-12:
            S.append(s)
            Yh.append(yv)
            if len(S) > mem:
                S.pop(0)
                Yh.pop(0)
        if rel < tol or np.abs(g).max() < 1e-8:
            break
    return {"mean": mean, "std": std, "flag_cols": flag_cols,
            "W": theta[:q * K].reshape(q, K).copy(), "b": theta[q * K:].copy()}


def _block(model, names, n):
    """The text between BEGIN FITTED and END FITTED. repr() of a float is exact, so the numbers
    read back are the fitted ones bit for bit."""
    def flat(name, values, fmt):
        return ["{} = [".format(name)] + ["    {},".format(fmt(v)) for v in values] + ["]"]
    num = lambda v: repr(float(v))
    lines = [BEGIN, "N_TRAIN = {}".format(int(n))]
    lines += flat("FEATURE_NAMES", names, repr)
    lines += flat("MEAN", model["mean"], num)
    lines += flat("STD", model["std"], num)
    lines += flat("FLAG_COLS", model["flag_cols"], lambda v: str(int(v)))
    lines += flat("W", model["W"], lambda r: "[" + ", ".join(num(v) for v in r) + "]")
    lines += flat("B", model["b"], num)
    lines.append(END)
    return "\n".join(lines)


def write_model(model, names, n, path=MODEL_FILE):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    a, b = text.index(BEGIN), text.index(END) + len(END)
    new = text[:a] + _block(model, names, n) + text[b:]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)


def main():
    t0 = time.time()
    X, y, names = build_sample()
    model = fit(X, y)
    write_model(model, names, len(y))
    print("written {} ({} rows, {} inputs, {:.0f} s)".format(
        os.path.relpath(MODEL_FILE, ROOT), len(y), len(names), time.time() - t0))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print("FAILED: {} at {}:{}".format(type(exc).__name__, os.path.basename(tb.filename), tb.lineno))
        sys.exit(1)
