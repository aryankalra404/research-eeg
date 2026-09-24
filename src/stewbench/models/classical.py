"""Feature-based and Riemannian-geometry baselines (scikit-learn/pyRiemann).

Hyper-parameters are selected on the fold's inner-validation subjects only
(fit on inner-train, score on inner-val, refit on inner-train), mirroring
checkpoint selection for the deep models.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features import spectral_features


def _covariances(X: np.ndarray) -> np.ndarray:
    from pyriemann.estimation import Covariances
    return Covariances(estimator="oas").fit_transform(X.astype(np.float64))


def _recenter(covariances: np.ndarray, subject: np.ndarray) -> np.ndarray:
    """Riemannian re-centering (Zanini et al., IEEE TBME 2018): whiten each
    subject's covariances by that subject's Riemannian mean (unlabelled)."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
    out = np.empty_like(covariances)
    for sid in np.unique(subject):
        mask = subject == sid
        whitening = invsqrtm(mean_riemann(covariances[mask]))
        out[mask] = whitening @ covariances[mask] @ whitening
    return out


class ClassicalModel:
    """Wraps a scikit-learn estimator plus its input representation."""

    def __init__(self, representation: str, estimator, grid: dict | None = None,
                 recenter: bool = False, seed: int = 0):
        self.representation = representation  # "spectral" | "covariance"
        self.estimator = estimator
        self.grid = grid or {}
        self.recenter = recenter
        self.seed = seed
        self.best_params_: dict = {}
        self.feature_names_: list[str] | None = None

    def transform(self, X: np.ndarray, subject: np.ndarray) -> np.ndarray:
        if self.representation == "spectral":
            features, self.feature_names_ = spectral_features(X)
            return features
        covariances = _covariances(X)
        return _recenter(covariances, subject) if self.recenter else covariances

    def fit(self, X, y, subject, X_val=None, y_val=None, subject_val=None, tune: bool = True):
        Z = self.transform(X, subject)
        candidates = [dict(zip(self.grid, values)) for values in itertools.product(*self.grid.values())]
        if tune and len(candidates) > 1 and X_val is not None:
            Z_val = self.transform(X_val, subject_val)
            scores = []
            for params in candidates:
                model = clone(self.estimator).set_params(**params).fit(Z, y)
                scores.append(log_loss(y_val, np.clip(model.predict_proba(Z_val), 1e-7, 1 - 1e-7), labels=[0, 1]))
            self.best_params_ = candidates[int(np.argmin(scores))]
        elif candidates:
            self.best_params_ = candidates[0]
        self.model_ = clone(self.estimator).set_params(**self.best_params_).fit(Z, y)
        return self

    def predict_proba(self, X, subject) -> np.ndarray:
        return self.model_.predict_proba(self.transform(X, subject))

    def parameter_count(self) -> int | None:
        est = self.model_[-1] if hasattr(self.model_, "steps") else self.model_
        for attr in ("coef_", "covmeans_"):
            if hasattr(est, attr):
                return int(np.size(getattr(est, attr)) + np.size(getattr(est, "intercept_", 0)))
        return None


def _scaled(estimator):
    return make_pipeline(StandardScaler(), estimator)


def build_classical(name: str, seed: int = 0) -> ClassicalModel:
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace

    C_grid = [0.01, 0.1, 1.0, 10.0]
    if name == "bp_lda":
        return ClassicalModel("spectral", _scaled(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))
    if name == "bp_lr":
        return ClassicalModel("spectral", _scaled(LogisticRegression(max_iter=5000)),
                              {"logisticregression__C": C_grid})
    if name == "bp_svm":
        # Platt-scaled probabilities via cross-validation (replaces the deprecated
        # SVC(probability=True), removed in scikit-learn 1.11).
        svm = CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), method="sigmoid", cv=5, ensemble=False)
        return ClassicalModel("spectral", _scaled(svm),
                              {"calibratedclassifiercv__estimator__C": [0.1, 1.0, 10.0],
                               "calibratedclassifiercv__estimator__gamma": ["scale", 0.001]})
    if name == "bp_rf":
        return ClassicalModel("spectral", RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=seed), {"min_samples_leaf": [1, 5, 20]})
    if name == "bp_gbdt":
        return ClassicalModel("spectral", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed), {"max_depth": [3, None], "l2_regularization": [0.0, 1.0]})
    if name == "riemann_mdm":
        return ClassicalModel("covariance", MDM(metric="riemann"))
    if name == "riemann_ts_lr":
        return ClassicalModel("covariance", make_pipeline(
            TangentSpace(metric="riemann"), StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": C_grid})
    if name == "riemann_rpa_ts_lr":
        return ClassicalModel("covariance", make_pipeline(
            TangentSpace(metric="riemann"), StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": C_grid}, recenter=True)
    raise KeyError(name)
