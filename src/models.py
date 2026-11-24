"""
Baseline and proposed models.

Baseline (population model):
    StandardScaler + LogisticRegression on raw window features. One model for
    everybody; it never sees anything about the new user.

Proposed (personal-baseline model):
    1. Personal calibration: each user's features are z-scored against their OWN
       first `calib_min` minutes of seated rest. This needs no stress labels from
       the user - only a short, consented "sit quietly" calibration that a watch
       app could ask for at setup.
    2. Classifier chosen by nested, subject-grouped cross-validation on the
       training people only (logistic regression vs. random forest).
    3. Probability calibration (sigmoid, subject-grouped folds) so the confidence
       score means something, which the abstention rule depends on.
    4. Abstention: if the calibrated probability is inside [lo, hi] the system
       says "not sure" instead of labelling the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def calibration_mask(df: pd.DataFrame, calib_min: float) -> pd.Series:
    """Windows that fall inside each subject's first `calib_min` minutes of baseline rest."""
    base = df[df["label"] == 1]
    first = base.groupby("subject")["t_start"].transform("min")
    in_calib = (base["t_start"] - first + 60) <= calib_min * 60
    mask = pd.Series(False, index=df.index)
    mask.loc[base.index[in_calib.values]] = True
    return mask


def personal_normalise(df: pd.DataFrame, feats: list[str], calib_min: float) -> pd.DataFrame:
    """z-score each subject's features using only their calibration windows."""
    out = df.copy()
    out[feats] = out[feats].astype(float)
    if calib_min <= 0:
        return out
    cm = calibration_mask(df, calib_min)
    for sid, idx in df.groupby("subject").groups.items():
        ref = df.loc[idx][cm.loc[idx]]
        mu = ref[feats].mean()
        sd = ref[feats].std().replace(0, np.nan).fillna(1.0)
        # guard against tiny std from only a couple of calibration windows
        sd = np.maximum(sd, df[feats].std() * 0.05)
        out.loc[idx, feats] = (df.loc[idx, feats] - mu) / sd
    return out


def make_baseline():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))


class PersonalStressModel(BaseEstimator, ClassifierMixin):
    """Proposed classifier, used on personally-normalised features.

    Inside each training fold it:
      1. picks between a logistic regression and a random forest with an inner
         GroupKFold over TRAINING subjects only (no peeking at the test person),
      2. refits the winner wrapped in sigmoid calibration, again with subject-grouped
         folds, so P(stress) is an honest confidence for the abstention rule.
    """
    needs_groups = True

    def __init__(self, inner_folds: int = 3):
        self.inner_folds = inner_folds

    @staticmethod
    def candidates():
        return {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", C=0.5)),
            "random_forest": RandomForestClassifier(
                n_estimators=150, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        }

    def fit(self, X, y, groups):
        gkf = GroupKFold(n_splits=self.inner_folds)
        splits = list(gkf.split(X, y, groups))
        self.inner_scores_ = {}
        for name, est in self.candidates().items():
            sc = []
            for tr, va in splits:
                m = clone(est).fit(X[tr], y[tr])
                sc.append(f1_score(y[va], m.predict(X[va]), average="macro"))
            self.inner_scores_[name] = float(np.mean(sc))
        self.chosen_ = max(self.inner_scores_, key=self.inner_scores_.get)
        self.model_ = CalibratedClassifierCV(self.candidates()[self.chosen_],
                                             method="sigmoid", cv=splits)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_proposed():
    return PersonalStressModel()


def make_rf_cal():
    """Ablation: calibrated random forest (used with and without personal norm)."""
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features="sqrt",
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=SEED)
    return CalibratedClassifierCV(rf, method="sigmoid", cv=3)


def make_lr_norm():
    """Ablation: the baseline classifier WITH personal normalisation."""
    return make_baseline()


def abstain(p: np.ndarray, lo: float = 0.35, hi: float = 0.65) -> np.ndarray:
    """Boolean mask: True where the system abstains."""
    return (p > lo) & (p < hi)

