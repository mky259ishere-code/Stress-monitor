"""
Leave-one-subject-out (LOSO) evaluation and metrics.

LOSO is the only honest split for this problem: windows from the same person are
highly correlated, so a random split would leak identity and inflate scores.
Every reported number is for a person the model has never seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)

from models import calibration_mask, personal_normalise


def loso_predict(df: pd.DataFrame, feats: list[str], model_fn, calib_min: float = 0.0,
                 drop_calibration_windows: bool = True) -> pd.DataFrame:
    """Return a copy of df's test rows with columns p_stress (probability) and fold."""
    data = personal_normalise(df, feats, calib_min)
    cm = calibration_mask(df, calib_min) if calib_min > 0 else pd.Series(False, index=df.index)
    preds = []
    for sid in df["subject"].unique():
        tr = data["subject"] != sid
        te = (data["subject"] == sid)
        if drop_calibration_windows:
            te &= ~cm                                  # never score the calibration minutes
        model = model_fn()
        Xtr, ytr = data.loc[tr, feats].values, data.loc[tr, "stress"].values
        if getattr(model, "needs_groups", False):
            model.fit(Xtr, ytr, groups=data.loc[tr, "subject"].values)
        else:
            model.fit(Xtr, ytr)
        out = df.loc[te, ["subject", "t_start", "label", "stress", "sex", "age"]].copy()
        out["p_stress"] = model.predict_proba(data.loc[te, feats].values)[:, 1]
        out["chosen_model"] = getattr(model, "chosen_", type(model).__name__)
        preds.append(out)
    return pd.concat(preds)


def ece(y, p, n_bins=10):
    """Expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    out = {
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "stress_f1": f1_score(y, yhat, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, yhat),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "ece": ece(y, p),
        "brier": brier_score_loss(y, p),
        "false_alarm_rate": float(((yhat == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "miss_rate": float(((yhat == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }
    return {k: float(v) for k, v in out.items()}


def per_subject(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in pred.groupby("subject", sort=False):
        m = metrics(g["stress"].values, g["p_stress"].values)
        m.update(subject=sid, sex=g["sex"].iloc[0], age=g["age"].iloc[0], n=len(g))
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_ci(per_subj: pd.DataFrame, col: str, n_boot=2000, seed=0):
    """95 % CI for the mean of a per-subject metric, resampling subjects."""
    rng = np.random.default_rng(seed)
    v = per_subj[col].dropna().values
    boots = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

