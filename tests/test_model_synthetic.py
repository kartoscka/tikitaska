"""Synthetic check of the whole pipeline: features -> rule_learnt_ensemble -> submission CSV.
Made-up match only (tests/synthetic_match.py): reads no match data.

Run:  python tests/test_model_synthetic.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import synthetic_match as S  # noqa: E402
from src import features as F  # noqa: E402
from src.models import extended_rules, rule_learnt_ensemble as M  # noqa: E402

RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def main():
    inp = S.make_inputs(seed=0)
    df = F.compute_features(inp["ball"], inp["players"], inp["team_of_slot"], inp["half"], inp["matrices"],
                            inp["hom_valid"], inp["hom_error"], inp["events"], inp["home_name"])
    labels = np.asarray(M.predict(df, inputs=inp)).astype(str)
    half = df["half"].to_numpy()

    check("one label per frame", len(labels) == len(df))
    check("only H, A, D", set(np.unique(labels)) <= {"H", "A", "D"})
    check("D outside the halves", (labels[half == 0] == "D").all())
    check("both teams get the ball", {"H", "A"} <= set(labels[half > 0]))

    P = M.probabilities(df, inputs=inp)
    check("probabilities sum to 1", np.allclose(P.sum(axis=1), 1.0))
    check("deterministic", (np.asarray(M.predict(df, inputs=inp)).astype(str) == labels).all())

    # with the learnt vote switched off the model is exactly the extended rules
    w = M.VOTE_WEIGHT
    try:
        M.VOTE_WEIGHT = 0.0
        no_vote = np.asarray(M.predict(df, inputs=inp)).astype(str)
    finally:
        M.VOTE_WEIGHT = w
    check("vote off = extended rules", (no_vote == np.asarray(extended_rules.predict(df)).astype(str)).all())

    from src.make_submission import write_csv
    path = os.path.join(tempfile.mkdtemp(), "sub.csv")
    full = np.full(199319, "D")
    full[:len(labels)] = labels
    write_csv(path, full)
    back = pd.read_csv(path)
    check("csv header and rows", list(back.columns) == ["frame", "label"] and len(back) == 199319)
    check("csv frames in order", (back["frame"].to_numpy() == np.arange(199319)).all())

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("ALL PASS" if n_fail == 0 else "FAILURES: {}".format(n_fail))
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
