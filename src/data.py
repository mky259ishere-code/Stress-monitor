"""
Data loading for wearable stress monitoring.

Two sources produce the SAME structure, so the rest of the pipeline does not care
which one is used:

    subject = {
        "id": "S2",
        "meta": {"sex": "M"/"F", "age": int},
        "signals": {name: (np.ndarray, fs_hz)},   # wrist only: EDA, TEMP, BVP, ACC
        "labels": (np.ndarray, fs_hz),            # 0 transient, 1 baseline, 2 stress,
                                                  # 3 amusement, 4 meditation
    }

1. load_wesad(root)       - the public WESAD dataset (Schmidt et al., 2018), wrist
                            (Empatica E4) channels only. Download from
                            https://archive.ics.uci.edu/dataset/465/wesad
                            and unzip so that root/S2/S2.pkl exists.
2. make_synthetic(...)    - a WESAD-format synthetic dataset with realistic
                            physiology and strong between-person differences.
                            Used when WESAD is not available, and for unit tests.

Only wrist signals are used on purpose (minimum-data design): a consumer
watch could collect them, the chest strap could not be worn all day.
"""
from __future__ import annotations

import os
import pickle
import re

import numpy as np

LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}

# WESAD E4 wrist sampling rates
FS = {"EDA": 4, "TEMP": 4, "BVP": 64, "ACC": 32}
LABEL_FS = 700


# --------------------------------------------------------------------------- #
# Real WESAD
# --------------------------------------------------------------------------- #
def _read_readme(path: str) -> dict:
    """Parse age and gender from WESAD's per-subject SX_readme.txt."""
    meta = {"sex": "unknown", "age": None}
    if not os.path.exists(path):
        return meta
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"Age:\s*(\d+)", txt)
    if m:
        meta["age"] = int(m.group(1))
    m = re.search(r"Gender:\s*(\w+)", txt)
    if m:
        meta["sex"] = "F" if m.group(1).lower().startswith("f") else "M"
    return meta


def load_wesad(root: str) -> list[dict]:
    if not root:
        raise ValueError(
            "WESAD root is required. Pass --wesad-root data/WESAD or set WESAD_ROOT."
        )
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"WESAD root does not exist or is not a directory: {root!r}. "
            "Expected folders such as S2/S2.pkl."
        )
    subjects = []
    for sid in sorted(os.listdir(root), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999):
        pkl = os.path.join(root, sid, f"{sid}.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        wrist = d["signal"]["wrist"]
        signals = {
            "EDA": (np.asarray(wrist["EDA"]).ravel(), FS["EDA"]),
            "TEMP": (np.asarray(wrist["TEMP"]).ravel(), FS["TEMP"]),
            "BVP": (np.asarray(wrist["BVP"]).ravel(), FS["BVP"]),
            "ACC": (np.asarray(wrist["ACC"], dtype=float) / 64.0, FS["ACC"]),  # to g
        }
        subjects.append({
            "id": sid,
            "meta": _read_readme(os.path.join(root, sid, f"{sid}_readme.txt")),
            "signals": signals,
            "labels": (np.asarray(d["label"]).ravel(), LABEL_FS),
        })
    if not subjects:
        raise FileNotFoundError(f"No WESAD subjects found under {root!r}")
    return subjects


# --------------------------------------------------------------------------- #
# Synthetic WESAD-format data
# --------------------------------------------------------------------------- #
# Protocol order and durations (minutes) roughly follow WESAD versions A/B.
PROTOCOL_A = [(1, 20), (0, 3), (3, 6.5), (0, 3), (4, 7), (0, 5), (2, 10), (0, 5), (4, 7)]
PROTOCOL_B = [(1, 20), (0, 3), (2, 10), (0, 5), (4, 7), (0, 3), (3, 6.5), (0, 3), (4, 7)]


def _scr_train(n, fs, rate_per_min, amp, rng):
    """Skin conductance responses: Poisson events convolved with a bi-exponential."""
    t = np.arange(int(fs * 20)) / fs
    kernel = np.exp(-t / 4.0) - np.exp(-t / 0.75)
    kernel /= kernel.max()
    events = rng.random(n) < rate_per_min / 60.0 / fs
    impulses = events * amp * rng.lognormal(0, 0.4, n)
    return np.convolve(impulses, kernel)[:n]


def _subject_synthetic(idx: int, rng: np.random.Generator) -> dict:
    sex = "F" if idx % 3 == 0 else "M"               # WESAD is ~80% male
    age = int(rng.integers(21, 36))
    protocol = PROTOCOL_A if idx % 2 == 0 else PROTOCOL_B

    # ---- person-specific physiology (the core difficulty of cross-person stress) --
    hr0 = rng.uniform(60, 88)
    eda0 = rng.lognormal(np.log(1.8), 0.8)           # 0.4 - 8 uS, huge spread
    temp0 = rng.uniform(31.0, 34.5)
    responder = rng.uniform(0.3, 1.2)                # how strongly this person reacts to stress
    if idx in (4, 11):                               # two weak responders (exist in WESAD too)
        responder = 0.2
    laugh = rng.uniform(0.1, 0.8)                    # how aroused the funny clips make them
    scr_amp = rng.uniform(0.05, 0.4) * (1 + eda0 / 3)
    m0 = rng.uniform(0.02, 0.07)                     # personal fidgetiness
    hrv0 = rng.uniform(0.012, 0.035)                 # personal beat-to-beat variability

    # condition effects: (dHR bpm, EDA level gain, SCR/min, dTEMP, motion multiplier, HRV mult)
    # Every effect is person-specific, and amusement overlaps stress on purpose.
    eff = {
        0: (2, 0.05, 2, 0.0, 1.3, 1.0),
        1: (0, 0.00, 1.5, 0.0, 1.0, 1.0),
        2: (responder * rng.normal(12, 4), responder * rng.uniform(0.2, 0.8),
            1.5 + 4 * responder * rng.uniform(0.5, 1.5), -0.3 * responder * rng.uniform(0, 1.5),
            rng.uniform(1.0, 2.0), 1 - 0.5 * responder * rng.uniform(0.3, 1)),
        3: (12 * laugh, 0.5 * laugh, 1.5 + 5 * laugh, 0.0, rng.uniform(1.2, 2.2), 1 - 0.2 * laugh),
        4: (-rng.uniform(0, 4), -0.05, 0.8, 0.05, 0.8, 1.1),
    }

    # build a 1 Hz state timeline
    seg_labels = []
    for lab, mins in protocol:
        seg_labels += [lab] * int(mins * 60 * rng.uniform(0.92, 1.08))
    lab_1hz = np.array(seg_labels)
    T = len(lab_1hz)

    def smooth_target(values, tau_s):
        out = np.empty_like(values, dtype=float)
        a = 1 / tau_s
        cur = values[0]
        for i, v in enumerate(values):
            cur += a * (v - cur)
            out[i] = cur
        return out

    dhr = smooth_target(np.array([eff[l][0] for l in lab_1hz]), 25)
    deda = smooth_target(np.array([eff[l][1] for l in lab_1hz]), 60)
    dtmp = smooth_target(np.array([eff[l][3] for l in lab_1hz]), 120)
    # motion: personal level x condition x slowly varying random fidgeting
    fidget = np.exp(np.repeat(np.cumsum(rng.normal(0, 0.25, T // 60 + 1)), 60)[:T])
    fidget /= fidget.mean()
    mot = m0 * np.array([eff[l][4] for l in lab_1hz]) * fidget
    hrv_mult = np.array([eff[l][5] for l in lab_1hz])
    scr_rate = np.array([eff[l][2] for l in lab_1hz])

    # ---- EDA (4 Hz) ------------------------------------------------------------
    fs = FS["EDA"]
    up = lambda x, f: np.repeat(x, f)
    n = T * fs
    drift = np.cumsum(rng.normal(0, 0.0015 * eda0, n))
    tonic = eda0 * (1 + up(deda, fs)) + drift
    phasic = np.zeros(n)
    rate_up = up(scr_rate, fs)
    for r in np.unique(rate_up):                     # piecewise-constant SCR rate
        mask = rate_up == r
        phasic += _scr_train(n, fs, r, scr_amp, rng) * mask
    eda = np.clip(tonic + phasic + rng.normal(0, 0.01, n), 0.01, None)

    # ---- TEMP (4 Hz) -----------------------------------------------------------
    temp = temp0 + up(dtmp, fs) + np.cumsum(rng.normal(0, 0.002, n)) + rng.normal(0, 0.02, n)

    # ---- BVP (64 Hz) from an instantaneous heart rate --------------------------
    fsb = FS["BVP"]
    hr = hr0 + dhr + 3 * np.sin(np.arange(T) / 40 * 2 * np.pi) * rng.uniform(0.3, 1)
    hr += rng.normal(0, 1.5, T) + np.cumsum(rng.normal(0, 0.15, T))   # slow drift
    hr_b = np.interp(np.arange(T * fsb) / fsb, np.arange(T), hr)
    # HRV: less beat-to-beat variability under stress
    hrv_jitter = up(hrv0 * hrv_mult, fsb) * rng.normal(0, 1, T * fsb)
    phase = np.cumsum(hr_b / 60 / fsb * (1 + hrv_jitter))
    bvp = 40 * (np.sin(2 * np.pi * phase) + 0.3 * np.sin(4 * np.pi * phase + 0.8))
    motion = up(mot, fsb)
    bvp += rng.normal(0, 1, len(bvp)) * (3 + 60 * motion)   # motion artefacts

    # ---- ACC (32 Hz, in g) -----------------------------------------------------
    fsa = FS["ACC"]
    g = np.array([0.1, -0.3, 0.95])
    acc = g + rng.normal(0, 1, (T * fsa, 3)) * up(mot, fsa)[:, None]

    labels_4hz = up(lab_1hz, 4)
    return {
        "id": f"S{idx + 2}",
        "meta": {"sex": sex, "age": age, "responder": round(float(responder), 2),
                 "amusement_arousal": round(float(laugh), 2)},
        "signals": {"EDA": (eda, fs), "TEMP": (temp, fs), "BVP": (bvp, fsb), "ACC": (acc, fsa)},
        "labels": (labels_4hz, 4),
    }


def make_synthetic(n_subjects: int = 15, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [_subject_synthetic(i, rng) for i in range(n_subjects)]


def load(source: str = "synthetic", wesad_root: str | None = None, seed: int = 7) -> list[dict]:
    if source == "wesad":
        return load_wesad(wesad_root or os.environ.get("WESAD_ROOT", "data/WESAD"))
    return make_synthetic(seed=seed)

