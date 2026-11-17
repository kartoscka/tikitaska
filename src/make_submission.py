"""Build the submission file from the raw match data, in one command.

    python -m src.make_submission                       # writes reports/submission/tikitaska.csv
    python -m src.make_submission --out path/to/file.csv
    python -m src.make_submission --rebuild-features    # recompute the feature table first

Steps:
  1. Feature table: reports/features.parquet (src/features.py). Built from the raw data if it is
     missing or --rebuild-features is given.
  2. Labels: src/models/rule_learnt_ensemble.predict, one of H / A / D per tactical video frame.
     The model also reads the raw tracking once (team widths and player movement).
  3. The CSV (frame,label), then the official validator (evaluation/validate_submission.py).

The data is found as the challenge's games.py finds it: data/ in the repo, or $MTAH_DATA_DIR.
The output is deterministic: the same data gives the same file, byte for byte.
"""

import argparse
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from src import features as F  # noqa: E402
from src.models import rule_learnt_ensemble as model  # noqa: E402

FEATURES = os.path.join(ROOT, "reports", "features.parquet")
DEFAULT_OUT = os.path.join(ROOT, "reports", "submission", "tikitaska.csv")


def build_features(path=FEATURES):
    """Compute the per-frame feature table from the raw data and write it to `path`."""
    F.main(["--out", os.path.relpath(path, ROOT)])


def write_csv(path, labels):
    """frame,label, one row per tactical video frame, '\\n' line ends on every platform."""
    n = GAMES[GAME]["n_video_frames"]
    labels = np.asarray(labels).astype(str)
    assert len(labels) == n, "need one label per video frame"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"frame": np.arange(n), "label": labels}).to_csv(path, index=False, lineterminator="\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the possession submission (frame,label CSV).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="CSV path (default reports/submission/tikitaska.csv)")
    ap.add_argument("--rebuild-features", action="store_true", help="recompute reports/features.parquet")
    ap.add_argument("--no-validate", action="store_true", help="skip the official validator")
    args = ap.parse_args(argv)

    if args.rebuild_features or not os.path.exists(FEATURES):
        build_features()
    df = pd.read_parquet(FEATURES)
    labels = model.predict(df)
    out = write_csv(args.out, labels)
    print("submission written: {}".format(os.path.relpath(out, ROOT)))
    if not args.no_validate:
        cmd = [sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out]
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        # Type and location only: an exception message can quote data.
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print("FAILED: {} at {}:{}".format(type(exc).__name__, os.path.basename(tb.filename), tb.lineno))
        sys.exit(1)
