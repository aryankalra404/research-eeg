"""Shared fold runner used by the benchmark and the augmentation study."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .. import constants as C
from ..data.inputs import model_inputs
from ..data.preprocess import WindowSet
from ..models import build_classical, spec
from ..settings import ExperimentConfig
from ..splits import Fold, masks
from ..training import OnlineAugment, predict_proba, train_deep


@dataclass
class PreparedData:
    windows: WindowSet   # filtered windows (device units) + labels + subjects
    X_input: np.ndarray  # normalized / aligned deep-model inputs, (N, C, T)

    @property
    def y(self):
        return self.windows.y

    @property
    def subject(self):
        return self.windows.subject


def prepare(windows: WindowSet, config: ExperimentConfig) -> PreparedData:
    return PreparedData(windows=windows, X_input=model_inputs(windows.X, windows.subject, config.input))


@dataclass
class FoldOutput:
    fold: int
    test_index: np.ndarray
    p1: np.ndarray
    best_epoch: int | None = None
    epochs_run: int | None = None
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    history: dict | None = None
    best_params: dict | None = None
    occlusion: dict = field(default_factory=dict)  # perturbation -> p1 on test windows
    model: object = None


def _bandstop(X: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    sos = butter(4, band, btype="bandstop", fs=C.SFREQ, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def perturbations(X_filtered: np.ndarray, seed: int):
    """Channel permutation importance (channel shuffled across test windows)
    and band ablation (4th-order band-stop), both applied to filtered data."""
    rng = np.random.default_rng(seed)
    for index, channel in enumerate(C.CHANNELS):
        X = X_filtered.copy()
        X[:, index] = X[rng.permutation(len(X)), index]
        yield f"channel:{channel}", X
    for name, band in C.FREQ_BANDS.items():
        yield f"band:{name}", _bandstop(X_filtered, band)


def run_fold(model_name: str, data: PreparedData, fold: Fold, seed: int, config: ExperimentConfig,
             device: torch.device, augment: OnlineAugment | None = None,
             X_extra: np.ndarray | None = None, y_extra: np.ndarray | None = None,
             interpret: bool = False, verbose: bool = False) -> FoldOutput:
    train, val, test = masks(data.subject, fold)
    y = data.y
    s = spec(model_name)
    if s.kind == "classical":
        if augment is not None or X_extra is not None:
            raise ValueError("Augmentation is only defined for deep models")
        X = data.windows.X
        model = build_classical(model_name, seed)
        t0 = time.perf_counter()
        model.fit(X[train], y[train], data.subject[train], X[val], y[val], data.subject[val],
                  tune=config.tune_classical)
        train_seconds = time.perf_counter() - t0
        predict = lambda Xf: model.predict_proba(Xf, data.subject[test])[:, 1]
        t0 = time.perf_counter()
        p1 = predict(X[test])
        out = FoldOutput(fold=fold.index, test_index=np.flatnonzero(test), p1=p1,
                         train_seconds=train_seconds, predict_seconds=time.perf_counter() - t0,
                         best_params=model.best_params_, model=model)
    else:
        training = config.training_for(model_name)
        result = train_deep(model_name, data.X_input[train], y[train], data.X_input[val], y[val],
                            training, seed, device, augment=augment, X_extra=X_extra, y_extra=y_extra,
                            sfreq=C.SFREQ, verbose=verbose)
        model = result.model

        def predict(Xf):
            X_in = model_inputs(Xf, data.subject[test], config.input)
            return predict_proba(model, X_in, device, amp=training.amp)[:, 1]
        t0 = time.perf_counter()
        p1 = predict_proba(model, data.X_input[test], device, amp=training.amp)[:, 1]
        out = FoldOutput(fold=fold.index, test_index=np.flatnonzero(test), p1=p1,
                         best_epoch=result.best_epoch, epochs_run=result.epochs_run,
                         train_seconds=result.train_seconds, predict_seconds=time.perf_counter() - t0,
                         history=result.history, model=model)
    if interpret:
        for name, X_perturbed in perturbations(data.windows.X[test], seed):
            out.occlusion[name] = predict(X_perturbed)
    return out
