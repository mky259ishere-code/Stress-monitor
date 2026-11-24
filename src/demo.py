"""
Live-style demo of the user-facing system (use this for the presentation video).

It replays one held-out person's session window by window, as a watch app would,
and prints exactly what the user would see. The model is trained on everyone
EXCEPT that person.

    python src/demo.py --subject S3 --consent
    python src/demo.py --subject S3 --consent --speed 0     # no delay

Responsible-design rules built into the loop:
  * Nothing runs without explicit consent (--consent). No consent -> exit.
  * 10-minute seated calibration first; until then the app only says it is learning.
  * Uncertainty-aware wording: never "you are stressed", always "your signals
    look similar to stress" with the confidence shown.
  * Abstains when 0.35 < P(stress) < 0.65.
  * Suggestions only, never actions: a 1-minute breathing exercise, dismissible.
  * Rate limit: at most one prompt per 10 minutes, so it cannot nag.
  * Nothing is stored: features exist in memory for one window and are dropped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from data import LABEL_NAMES, load                      # noqa: E402
from features import ALL_FEATURES, extract_all           # noqa: E402
from models import calibration_mask, make_proposed, personal_normalise  # noqa: E402

CALIB_MIN = 10
COOLDOWN_S = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S3")
    ap.add_argument("--source", choices=["synthetic", "wesad"], default="synthetic")
    ap.add_argument("--wesad-root", default=None)
    ap.add_argument("--consent", action="store_true", help="user agrees to on-device stress estimation")
    ap.add_argument("--speed", type=float, default=0.05, help="seconds of delay per window")
    args = ap.parse_args()

    if not args.consent:
        print("Stress check-ins are OFF. Nothing is being measured.\n"
              "Turn them on with --consent (you can turn them off at any time).")
        return

    df = extract_all(load(args.source, args.wesad_root))
    if args.subject not in set(df["subject"]):
        sys.exit(f"Unknown subject {args.subject}. Options: {sorted(df['subject'].unique())}")
    data = personal_normalise(df, ALL_FEATURES, CALIB_MIN)
    calib = calibration_mask(df, CALIB_MIN)
    tr = data["subject"] != args.subject
    model = make_proposed().fit(data.loc[tr, ALL_FEATURES].values, data.loc[tr, "stress"].values,
                                groups=data.loc[tr, "subject"].values)
    me = data[data["subject"] == args.subject].sort_values("t_start")

    print(f"Consent given. Model trained on {tr.sum()} windows from other people "
          f"(chose: {model.chosen_}).\n")
    last_prompt = -1e9
    for idx, row in me.iterrows():
        t = row["t_start"] / 60
        truth = LABEL_NAMES[int(df.loc[idx, "label"])]
        if calib.loc[idx]:
            msg = "Learning your resting pattern... (calibration)"
        else:
            p = model.predict_proba(row[ALL_FEATURES].values.reshape(1, -1).astype(float))[0, 1]
            if 0.35 < p < 0.65:
                msg = f"  -  not sure right now ({p:.0%})"
            elif p >= 0.65 and row["t_start"] - last_prompt >= COOLDOWN_S:
                msg = (f"Your signals look similar to stress ({p:.0%} confidence). "
                       "Want a 1-minute breathing exercise? [Start] [Not now] [Wrong]")
                last_prompt = row["t_start"]
            elif p >= 0.65:
                msg = f"  -  similar to stress ({p:.0%}), prompt suppressed (cooldown)"
            else:
                msg = f"  -  looks calm ({1 - p:.0%})"
        print(f"[{t:5.1f} min | truth: {truth:10s}] {msg}")
        time.sleep(args.speed)
    print("\nSession ended. No physiological data was saved.")


if __name__ == "__main__":
    main()

