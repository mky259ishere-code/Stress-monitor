"""
Sliding-window feature extraction from wrist signals.

Windows are 60 s long with a 30 s step. A window is kept only if >= 80 % of its
samples share one protocol label in {1,2,3,4}; transitional windows are dropped.

Features (20 total), all cheap enough to compute on a watch:
  EDA  : mean, std, min, max, slope, SCR count, SCR mean amplitude, phasic energy
  BVP  : mean HR, std HR, RMSSD (from detected pulse peaks), peak count quality
  TEMP : mean, std, slope
  ACC  : magnitude mean, magnitude std, per-axis mean std, activity fraction
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

WIN_S = 60
STEP_S = 30
PURITY = 0.8

FEATURE_GROUPS = {
    "EDA": ["eda_mean", "eda_std", "eda_min", "eda_max", "eda_slope",
            "scr_count", "scr_amp", "eda_phasic_energy"],
    "BVP": ["hr_mean", "hr_std", "hr_rmssd", "bvp_peak_quality"],
    "TEMP": ["temp_mean", "temp_std", "temp_slope"],
    "ACC": ["acc_mag_mean", "acc_mag_std", "acc_axis_std", "acc_active_frac", "acc_jerk"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _slope(x, fs):
    t = np.arange(len(x)) / fs
    return np.polyfit(t, x, 1)[0] if len(x) > 2 else 0.0


def _eda_feats(x, fs):
    # split tonic/phasic with a 0.05 Hz low-pass
    b, a = butter(2, 0.05 / (fs / 2), "low")
    tonic = filtfilt(b, a, x) if len(x) > 15 else x
    bs, as_ = butter(2, 1.0 / (fs / 2), "low")            # remove sensor noise first
    xs = filtfilt(bs, as_, x) if len(x) > 15 else x
    phasic = xs - tonic
    peaks, props = find_peaks(phasic, prominence=0.03, distance=int(fs * 1.5))
    return {
        "eda_mean": x.mean(), "eda_std": x.std(), "eda_min": x.min(), "eda_max": x.max(),
        "eda_slope": _slope(x, fs), "scr_count": len(peaks),
        "scr_amp": props["prominences"].mean() if len(peaks) else 0.0,
        "eda_phasic_energy": float(np.mean(phasic ** 2)),
    }


def _bvp_feats(x, fs):
    b, a = butter(2, [0.7 / (fs / 2), 3.5 / (fs / 2)], "band")
    y = filtfilt(b, a, x)
    peaks, _ = find_peaks(y, distance=int(fs * 0.33), prominence=np.std(y) * 0.5)
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]                 # 40-180 bpm
    if len(ibi) < 5:
        return {"hr_mean": np.nan, "hr_std": np.nan, "hr_rmssd": np.nan, "bvp_peak_quality": 0.0}
    hr = 60 / ibi
    return {
        "hr_mean": hr.mean(), "hr_std": hr.std(),
        "hr_rmssd": float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000),
        "bvp_peak_quality": len(ibi) / max(len(peaks) - 1, 1),
    }


def _temp_feats(x, fs):
    return {"temp_mean": x.mean(), "temp_std": x.std(), "temp_slope": _slope(x, fs)}


def _acc_feats(x, fs):
    mag = np.linalg.norm(x, axis=1)
    return {
        "acc_mag_mean": mag.mean(), "acc_mag_std": mag.std(),
        "acc_axis_std": x.std(axis=0).mean(),
        "acc_active_frac": float(np.mean(np.abs(mag - np.median(mag)) > 0.1)),
        "acc_jerk": float(np.mean(np.abs(np.diff(mag))) * fs),
    }


EXTRACTORS = {"EDA": _eda_feats, "BVP": _bvp_feats, "TEMP": _temp_feats, "ACC": _acc_feats}


def extract_subject(subj: dict) -> pd.DataFrame:
    labels, lfs = subj["labels"]
    duration = len(labels) / lfs
    for sig, fs in subj["signals"].values():
        duration = min(duration, len(sig) / fs)
    rows = []
    # Include a window that ends exactly at the available signal duration.
    # np.arange's exclusive stop would otherwise drop that final valid window.
    for k, start in enumerate(np.arange(0, duration - WIN_S + STEP_S, STEP_S)):
        lab = labels[int(start * lfs): int((start + WIN_S) * lfs)]
        vals, counts = np.unique(lab, return_counts=True)
        top = vals[counts.argmax()]
        if top not in (1, 2, 3, 4) or counts.max() / len(lab) < PURITY:
            continue
        row = {"subject": subj["id"], "t_start": start, "label": int(top),
               "sex": subj["meta"].get("sex", "unknown"), "age": subj["meta"].get("age")}
        for name, (sig, fs) in subj["signals"].items():
            seg = sig[int(start * fs): int((start + WIN_S) * fs)]
            row.update(EXTRACTORS[name](seg, fs))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["window_idx"] = np.arange(len(df))
    return df


def extract_all(subjects: list[dict]) -> pd.DataFrame:
    df = pd.concat([extract_subject(s) for s in subjects], ignore_index=True)
    # fill rare failed HR windows with the subject's median
    for c in ALL_FEATURES:
        df[c] = df.groupby("subject")[c].transform(lambda s: s.fillna(s.median()))
    df["stress"] = (df["label"] == 2).astype(int)
    return df

