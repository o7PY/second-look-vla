"""The retry-value model of Part 4 (SPEC_part4.md section 5): a T-learner = two calibrated classifiers f_retry(x), f_cont(x) = P(success | retry), P(success | continue) over a group-selected column subset of
`second_look.features.FEATURE_NAMES`. `SHModel.predict` is what the loop calls at the first alarm; the object is pickled (`configs/part4_model.pkl`, sha256 recorded)."""
from __future__ import annotations

import numpy as np

from second_look import features as F


class SHModel:
    def __init__(self, name: str, groups: str, kind: str, f_retry, f_cont):
        self.name, self.groups, self.kind = name, groups, kind
        self.cols = F.columns(groups)
        self.f_retry, self.f_cont = f_retry, f_cont

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        X = np.asarray(x, np.float64)[None, self.cols]
        return float(self.f_retry.predict_proba(X)[0, 1]), float(self.f_cont.predict_proba(X)[0, 1])

    def predict_many(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, np.float64)[:, self.cols]
        return self.f_retry.predict_proba(X)[:, 1], self.f_cont.predict_proba(X)[:, 1]


class Constant:
    """A degenerate classifier (a training fold with one class only)."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        return np.tile([1.0 - self.p, self.p], (len(X), 1))


class StubModel:
    """Tests: a fixed decision. predict() returns (p_retry, p_cont)."""

    def __init__(self, p_retry: float, p_cont: float):
        self.p = (p_retry, p_cont)

    def predict(self, x):
        return self.p


class GBTCalibrated:
    """Gradient-boosted trees (depth <= 3, <= 200 trees) with isotonic calibration, cross-fitted by GROUP (the branches of one alarm case share their features, so a plain stratified calibration split would leak):
    `fit(X, y, groups)`: the isotonic map is fitted on the out-of-group raw scores of a 3-fold grouped split, the final trees on all rows."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05, subsample: float = 0.8, seed: int = 0):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, random_state=seed)
        self.gbt = self.iso = None
        self.constant = None

    def fit(self, X, y, groups):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import GroupKFold
        X, y, groups = np.asarray(X, np.float64), np.asarray(y, int), np.asarray(groups)
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            return self
        raw = np.full(len(y), np.nan)
        if len(np.unique(groups)) >= 3:
            for tr, te in GroupKFold(n_splits=3).split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                raw[te] = GradientBoostingClassifier(**self.params).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        ok = ~np.isnan(raw)
        self.gbt = GradientBoostingClassifier(**self.params).fit(X, y)
        if ok.sum() >= 20 and len(np.unique(y[ok])) == 2:
            self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw[ok], y[ok])
        return self

    def predict_proba(self, X):
        if self.constant is not None:
            return Constant(self.constant).predict_proba(X)
        p = self.gbt.predict_proba(np.asarray(X, np.float64))[:, 1]
        if self.iso is not None:
            p = self.iso.predict(p)
        return np.stack([1.0 - p, p], axis=1)

    @property
    def feature_importances_(self):
        return None if self.gbt is None else self.gbt.feature_importances_


class Logistic:
    """L2-logistic regression on standardised features (C = 1); a degenerate training set (one class) gives a constant."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.pipe = None
        self.constant = None

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        y = np.asarray(y, int)
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            return self
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=5000)).fit(np.asarray(X, np.float64), y)
        return self

    def predict_proba(self, X):
        if self.constant is not None:
            return Constant(self.constant).predict_proba(X)
        return self.pipe.predict_proba(np.asarray(X, np.float64))

    @property
    def coef_(self):
        return None if self.pipe is None else self.pipe[-1].coef_[0]


# Compatibility: configs/part4_model.pkl was pickled when this package was called `sl` (class path `sl.shmodel.SHModel`). Its sha256 is frozen
# (configs/part4_model.json), so the file is not re-pickled; instead the old module path is aliased to this module.
import sys as _sys  # noqa: E402
_sys.modules.setdefault("sl", _sys.modules[__name__.split(".")[0]])
_sys.modules["sl.shmodel"] = _sys.modules[__name__]
