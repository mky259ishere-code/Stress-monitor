# Wearable-Based Stress Monitoring with Personal Baselines

CSBP 499 Affective Computing, semester project.

**Research question.** Can a wrist wearable tell, for a person it has never seen
before, whether their body is in a stress-like state, and does a short, label-free
personal calibration (10 minutes of seated rest) make that more accurate and
better-calibrated than a one-size-fits-all model?

**Intended user benefit.** An opt-in watch feature that notices stress-like
physiology and *offers* a 1-minute breathing exercise, with honest confidence and
a "not sure" option. It supports the user; it does not diagnose them.

## Repository layout

```
src/
  data.py       WESAD loader (wrist only) + WESAD-format synthetic generator
  features.py   60 s windows, 30 s step; 20 EDA / BVP / TEMP / ACC features
  models.py     baseline, ablations, proposed model, personal normalisation, abstention
  evaluate.py   leave-one-subject-out evaluation, metrics, bootstrap CIs
  run.py        runs experiments E1-E7, writes results/ (JSON, CSV, figures)
  demo.py       replays one held-out person as a watch app would (for the video)
results/        produced by run.py
requirements.txt
```

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce every number in the report

```bash
python src/run.py                     # synthetic WESAD-format data, ~5 min on 2 CPU cores
```

Everything is seeded (data seed 7, model seed 42), so the same command gives the
same `results/results.json`.

### Run on the real WESAD dataset

1. Download WESAD (free for non-commercial research) from the UCI repository:
   https://archive.ics.uci.edu/dataset/465/wesad
2. Unzip so that `data/WESAD/S2/S2.pkl` exists (folders S2 ... S17; S1 and S12 are absent).
3. Run:

```bash
python src/run.py --source wesad --wesad-root data/WESAD --out results_wesad
python src/demo.py --source wesad --wesad-root data/WESAD --subject S5 --consent
```

Loading the pickles needs ~6 GB RAM; feature extraction takes a few minutes.

## Demo

```bash
python src/demo.py --subject S3                # no consent -> nothing is measured
python src/demo.py --subject S3 --consent      # replay a held-out person
```

## Method in one paragraph

Wrist signals (EDA 4 Hz, skin temperature 4 Hz, BVP 64 Hz, accelerometer 32 Hz)
are cut into 60 s windows. Windows with less than 80 % of one protocol label are
dropped. Task: stress vs. not-stress (baseline, amusement and meditation are all
"not stress"). **Baseline:** standardised logistic regression trained on everyone
else. **Proposed:** (1) every person's features are z-scored against their own
first 10 minutes of rest, which needs no stress labels; (2) the classifier
(logistic regression or random forest) is picked by nested, subject-grouped
cross-validation on the training people; (3) its probabilities are sigmoid-calibrated;
(4) the system abstains when 0.35 < P(stress) < 0.65. All evaluation is
leave-one-subject-out, and the calibration minutes are never scored.

## About the synthetic data

The development environment could not download WESAD, so results in the report
were produced on a synthetic dataset in exactly the WESAD format (15 people,
same protocol, same sensors and sampling rates). It deliberately includes the
things that make real stress detection hard: large person-to-person differences
in resting physiology, weak responders, amusement that raises arousal like stress
does, motion artefacts, and slow drift. The code paths are identical for real
WESAD; re-run with `--source wesad` and report those numbers alongside.

