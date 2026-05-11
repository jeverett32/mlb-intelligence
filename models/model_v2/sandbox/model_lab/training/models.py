"""Model factories for sandbox training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def make_lr(C: float = 0.5, l1_ratio: float | None = None) -> Pipeline:
    """L2 (default) or elastic-net LR with median-impute + standard-scale."""
    if l1_ratio is None:
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=4000, random_state=42)
    else:
        clf = LogisticRegression(
            C=C,
            solver="saga",
            l1_ratio=l1_ratio,
            penalty="elasticnet",
            max_iter=4000,
            random_state=42,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=True)),
            ("model", clf),
        ]
    )


def make_early_lr() -> Pipeline:
    return make_lr(C=0.3)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class OffsetLogit:
    """Logistic regression with an additive logit offset.

    Model: p = sigmoid(offset + intercept + X @ w)
    Fit: L2-regularized MLE using scipy L-BFGS-B.

    Intended usage: offset = logit(p_market). We learn residual edge vs market.
    """

    w: np.ndarray
    intercept: float
    scaler: StandardScaler
    imputer: SimpleImputer

    def predict_proba(self, X_raw: np.ndarray, offset: np.ndarray) -> np.ndarray:
        X = self.imputer.transform(X_raw)
        X = self.scaler.transform(X)
        z = offset + self.intercept + X @ self.w
        p = _sigmoid(z)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def fit_offset_logit(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    offset: np.ndarray,
    l2: float = 1.0,
) -> OffsetLogit:
    """Fit offset logistic regression with L2 penalty."""
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X = imputer.fit_transform(X_raw)
    scaler = StandardScaler(with_mean=True)
    X = scaler.fit_transform(X)

    y = y.astype(float)
    offset = offset.astype(float)

    n, d = X.shape
    x0 = np.zeros(d + 1)  # [intercept, w...]

    def obj_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b0 = theta[0]
        w = theta[1:]
        z = offset + b0 + X @ w
        p = _sigmoid(z)
        # Negative log-likelihood
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # L2 penalty (do not penalize intercept)
        nll += 0.5 * l2 * float(np.dot(w, w))

        # Gradient
        r = (p - y)  # d(nll)/dz
        g0 = float(np.sum(r))
        gw = X.T @ r + l2 * w
        g = np.concatenate([[g0], gw])
        return float(nll), g

    def fun(theta: np.ndarray) -> float:
        v, _ = obj_grad(theta)
        return v

    def jac(theta: np.ndarray) -> np.ndarray:
        _, g = obj_grad(theta)
        return g

    res = minimize(fun, x0, method="L-BFGS-B", jac=jac, options={"maxiter": 400})
    theta = res.x if res.success else x0
    return OffsetLogit(w=theta[1:], intercept=float(theta[0]), scaler=scaler, imputer=imputer)


def make_lgbm(**overrides):
    if lgb is None:
        raise ImportError("lightgbm not installed")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "importance_type": "gain",
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    }
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


def make_xgb(**overrides):
    if xgb is None:
        raise ImportError("xgboost not installed")
    params = {
        "objective": "binary:logistic",
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    params.update(overrides)
    return xgb.XGBClassifier(**params)
