# Metrica sample games: possession labels

Possession labels for two other matches, used to train the learnt vote of
`rule_learnt_ensemble` (`src/training/`).

## The data

- **Source:** [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data),
  commit `e706dd5`. Published by Metrica Sports for the football analytics community. Their readme
  asks to credit the source when the data is used publicly: data by Metrica Sports.
- **What it is:** anonymised full matches (Sample_Game_1, Sample_Game_2), tracking at 25 fps plus an
  event log. Not BAR-ATH, and not part of the challenge bundle. The rules allow data about other
  games.
- **Where it goes:** the raw files are not in this repo. `python -m src.training.metrica_samples`
  downloads them into `data/external/metrica/` (git-ignored).

## The label files

`Sample_Game_<g>_labels.csv`: one row per spell of `H`, `A` or `D`.
`start_frame` is included, `end_frame` is not; frames are 0-based rows of the tracking files.

They are built from each game's event log by a fixed rule, not by hand and not by a model:

1. A pass, shot, recovery, set piece, or won duel or dribble gives the ball to that team.
2. "Ball lost" alone changes nothing: a loose ball stays with the last team until the other team
   records an event.
3. Ball out, a foul, a shot that ends out or in a goal, and the end of a half make the ball dead
   until the next restart (a set piece, or a goal-kick pass).
4. The challenge's definition: possession changes only when the other team gets control. A short
   spell of the other team (under 2 s, with no pass, shot or set piece) counts as a touch without
   control, and the ball stays with the previous team.

Rebuild them with `python -m src.training.metrica_samples --write-labels`.

## Limits

- The organisers build their labels from corrected tracking; these come from events. They are a
  close stand-in for the challenge's definition, not the same thing.
- The restart tags on these games come from the same event log as the labels. BAR-ATH's come from
  Smart Tagging, which misses some, so the training thins the tags to Smart Tagging's rate on
  BAR-ATH and moves them by up to 2 s.
