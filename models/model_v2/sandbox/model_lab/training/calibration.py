"""Calibration comparison: none vs Platt vs isotonic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationResult:
    method: str
    probs: np.ndarray


def fit_platt(probs_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(probs_train.reshape(-1, 1), y_train)
    return lr


def apply_platt(model: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    return model.predict_proba(probs.reshape(-1, 1))[:, 1]


def fit_isotonic(probs_train: np.ndarray, y_train: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_train, y_train)
    return iso


def calibrate_all(
    probs_train: np.ndarray,
    y_train: np.ndarray,
    probs_test: np.ndarray,
) -> dict[str, np.ndarray]:
    """Returns dict of method -> calibrated probs on test set."""
    out = {"none": probs_test}
    try:
        platt = fit_platt(probs_train, y_train)
        out["platt"] = np.clip(apply_platt(platt, probs_test), 1e-6, 1 - 1e-6)
    except Exception:
        pass
    try:
        iso = fit_isotonic(probs_train, y_train)
        out["isotonic"] = np.clip(iso.predict(probs_test), 1e-6, 1 - 1e-6)
    except Exception:
        pass
    return out


def wrap_calibrated_cv(estimator, method: str = "isotonic", cv: int = 3):
    """For tree models: wrap with sklearn CalibratedClassifierCV."""
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
