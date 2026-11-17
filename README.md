# Tikitaska: ball possession per frame

Team Tikitaska's entry to **More than a Hack 2027** (FC Barcelona vs Athletic Club, LaLiga 2024-25).
For every frame of the tactical video (199,319 frames, 25 fps) the pipeline outputs one label:
`H` (home in possession), `A` (away in possession) or `D` (dead ball).

The model is **`rule_learnt_ensemble`**: hand-designed football rules plus one learnt vote, decoded
into clean possession spells by a Viterbi pass with switching costs.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # Windows: .venv\Scripts\pip ...

# match data: unpack the challenge bundle into ./data, or point MTAH_DATA_DIR at it
#   data/tactical/BAR-ATH_tactical_atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml

python -m src.make_submission                         # -> reports/submission/tikitaska.csv
```

`make_submission` builds the per-frame feature table from the raw data (`reports/features.parquet`),
runs the model, writes the `frame,label` CSV and runs the official validator on it. Options:
`--out <path>`, `--rebuild-features`, `--no-validate`. The output is deterministic: the same data
gives the same file, byte for byte. Nothing derived from the match data is committed (`reports/`
and `data/` are git-ignored).

Tested with Python 3.14 and the pinned versions in `requirements.txt` (numpy, pandas, pyarrow).

## How it works

### 1. Features (`src/features.py`)
One row per tactical frame, built only from the automatic tracking (ATD), the homography and
Smart Tagging. The ball and every player track are cleaned (physically impossible jumps removed,
gaps under 1 s interpolated, local straight-line fits for position and velocity). Per frame: ball
position, speed, how long it has been missing, whether it is over a line; for each team the
nearest player's distance, players within 2 m and 5 m, mean speed, players off camera; the attacking
direction per half; and the active Smart Tagging codes. Every threshold is physical or football
reasoning and is documented in [docs/features.md](docs/features.md). None is fitted on this match.

### 2. Rule votes (`src/models/base_rules.py`, `src/models/extended_rules.py`)
Each rule gives a vote from 0 to 1 for `H`, `A` or `D` on every frame:

| rule | votes for | idea |
|---|---|---|
| reach | H / A | the nearest player of each team close to the ball (off-camera players count half) |
| pass2 | H / A | a kick followed by 5 m or more of ball travel: the ball belongs to the passer's team in flight |
| gap | D | the ball lost for a long time, only for gaps that start at a line, or when all players are almost still |
| out | D | ball over a line, or both teams almost stopped |
| restart | D, then H / A | Smart Tagging set pieces: dead ball up to the restart, then the taking team; after a goal, dead until the kick-off |
| phase | H / A | Smart Tagging phases of play (build up, progression, ...) vote, with a low weight, for the team in possession |

### 3. The learnt vote (`src/models/rule_learnt_ensemble.py`)
A multinomial logistic regression (numpy only) gives P(H), P(A), P(D) per frame from 70 inputs: the
rule votes, ball state, distances, team widths (the team with the ball opens the pitch), how the
players speed up, slow down and close on the ball, corner and goal-kick shapes, and 0.2 s / 1 s
moving averages and 1 s lags of some of these. The three probabilities are added to the rule votes
with weight 0.5. The fitted numbers are written in the model file.

### 4. Decoding
The weighted votes are summed and a Viterbi pass picks the best `H`/`A`/`D` path in each half under
switching costs (12 for `H`<->`A`, 8 into or out of `D`). Possession therefore changes only when the
other side wins clearly for some time, and a loose ball stays with the last team. Frames outside
the two halves are `D`.

## Training data (no data from this match's labels)

The learnt vote is fitted on **Metrica Sports' public sample games 1 and 2**
([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), commit `e706dd5`;
data by Metrica Sports). Their possession labels are built from the public event logs by a fixed
rule that follows the challenge's definition (see
[external/metrica_samples/README.md](external/metrica_samples/README.md)): no hand labels.

To look like BAR-ATH's tracking, the samples are degraded with four traits of BAR-ATH's tracking,
all measured without labels: the ball runs about 6 frames ahead (`LEAD_FRAMES`), with a slow drift
(`DRIFT_M`); 0 / 15 / 30 % of the live ball is deleted in gaps of BAR-ATH's own gap lengths; and the
restart tags are thinned to BAR-ATH's Smart Tagging rate (set pieces per minute) and moved by up to
2 s. No BAR-ATH label is used. The public 5-minute ground truth is not used for training, tuning or
model selection.

Rebuild the fitted numbers (downloads the sample games on first use; needs
`reports/features.parquet`):

```bash
python -m src.training.train_rule_learnt_ensemble    # rewrites the FITTED block
```

A refit is seeded, but it is not bit-identical across machines: numerical libraries differ in the
last digits, and two refits on different machines differed by up to 0.018 in single weights. That
moves 1 of 199,319 frames of the output. The submission uses the weights committed in
`src/models/rule_learnt_ensemble.py`; from those, `make_submission` gives the same file byte for byte.

## Tests (synthetic data only)

```bash
python tests/test_features_synthetic.py
python tests/test_model_synthetic.py
```

## Repository layout

```
src/
  make_submission.py               one-command pipeline: features -> model -> CSV -> validator
  features.py                      per-frame feature table from the raw data
  models/
    base_rules.py                  rule votes, Viterbi decoder
    extended_rules.py              extra rules (phases, restart types, slow-play dead ball)
    rule_learnt_ensemble.py        the submitted model: rules + learnt vote
  training/
    metrica_samples.py             training data from the public Metrica sample games
    train_rule_learnt_ensemble.py  refits the learnt vote
external/metrica_samples/          labels of the sample games (built from public event logs)
docs/features.md                   data dictionary of the feature table
tests/                             synthetic checks
games.py, starter/, evaluation/    challenge code, unchanged (upstream commit f8f2629)
```

## Known limits

- The ATD ball has no height and is missing on a large share of frames, most often where it
  matters (crowded boxes, aerial balls). While it is missing, possession comes from the players.
- Smart Tagging is automatic and has false positives and negatives; the phase votes are kept low.
- The training labels come from event logs of other matches, not from corrected tracking like the
  organisers' ground truth.

## License

MIT (see [LICENSE](LICENSE)). Challenge code in `games.py`, `starter/` and `evaluation/` is from
`metrica-sports/more-than-a-hack-2027`.
