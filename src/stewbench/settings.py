"""Typed experiment configuration loaded from YAML files in ``configs/``.

Every number that affects a reported result lives in one of these dataclasses
and is written verbatim into each run's manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PreprocessConfig:
    bandpass_hz: tuple[float, float] = (0.5, 45.0)
    filter_order: int = 4
    window_seconds: float = 4.0
    window_overlap: float = 0.5
    # Samples trimmed from the start/end of every recording (seconds). STEW
    # recordings contain set-up transients at the edges of some files.
    trim_seconds: float = 0.0
    # Reject windows whose peak-to-peak amplitude on any channel exceeds
    # median + k * MAD of that subject's windows (class-agnostic). None = off.
    artifact_mad_k: float | None = 60.0

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


@dataclass
class ProtocolConfig:
    # "group_kfold": K folds of whole subjects. "loso": leave-one-subject-out.
    scheme: str = "group_kfold"
    n_folds: int = 10
    # Fixes the subject-to-fold assignment. Deliberately separate from the
    # training seeds so every model/seed is evaluated on identical folds.
    fold_seed: int = 2024
    # Fraction of each fold's training subjects held out (subject-disjoint)
    # for early stopping / checkpoint selection. Never the test subjects.
    inner_val_fraction: float = 0.15
    # Optional cap on the number of training subjects per fold (data-efficiency
    # experiments). None = use all.
    max_train_subjects: int | None = None


@dataclass
class InputConfig:
    # Per-window, per-channel z-score ("window_zscore") uses no information
    # outside the window and is the strict default. "none" keeps microvolts.
    normalization: str = "window_zscore"
    # "euclidean" applies Euclidean Alignment (He & Wu, 2020) per subject using
    # that subject's *unlabelled* windows. It is transductive for the test
    # subject and must be reported as such.
    alignment: str = "none"


@dataclass
class TrainingConfig:
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # "cosine" | "none"
    warmup_epochs: int = 3
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = 1.0
    # Checkpoint-selection metric on the inner-validation subjects.
    monitor: str = "val_loss"  # "val_loss" | "val_balanced_accuracy"
    amp: bool = True  # automatic mixed precision on CUDA only
    deterministic: bool = True


@dataclass
class AugmentationConfig:
    # Name from stewbench.augment.transforms.ONLINE_AUGMENTATIONS, a generative
    # model ("cwgan_gp", "ddpm"), or "none".
    method: str = "none"
    # Online transforms: probability of applying the transform to a sample.
    probability: float = 0.5
    magnitude: float = 0.5
    # Offline (generative) methods: synthetic windows per class, as a fraction
    # of that class's real inner-training windows.
    synthetic_fraction: float = 1.0
    generator_epochs: int = 300
    generator_batch_size: int = 64


@dataclass
class ExperimentConfig:
    name: str = "benchmark"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    models: list[str] = field(default_factory=list)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    input: InputConfig = field(default_factory=InputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Per-model overrides of TrainingConfig fields, e.g. {"eeg_conformer": {"lr": 5e-4}}.
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Augmentation study only: list of AugmentationConfig-compatible dicts.
    augmentations: list[AugmentationConfig] = field(default_factory=list)
    # Augmentation study only: numbers of training subjects per fold for the
    # data-efficiency curve (null = all available). Subsets are nested.
    train_subject_budgets: list[int | None] = field(default_factory=lambda: [None])
    # Classical models: small grid search on inner folds of training subjects.
    tune_classical: bool = True
    # Model-agnostic interpretability (channel / band occlusion) on test folds.
    interpretability: bool = True
    save_checkpoints: bool = False
    num_threads: int | None = None

    def training_for(self, model_name: str) -> TrainingConfig:
        overrides = self.model_overrides.get(model_name, {})
        return _replace_dataclass(self.training, overrides)

    def to_dict(self) -> dict:
        return asdict(self)


def _replace_dataclass(instance, overrides: dict):
    values = asdict(instance)
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"Unknown {type(instance).__name__} fields: {sorted(unknown)}")
    values.update(overrides)
    return _build(type(instance), values)


def _build(cls, data: dict | None):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name) if name in known else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        elif name == "augmentations":
            kwargs[name] = [
                item if isinstance(item, AugmentationConfig) else _build(AugmentationConfig, item)
                for item in value
            ]
        elif name == "bandpass_hz":
            kwargs[name] = tuple(float(v) for v in value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    config = _build(ExperimentConfig, data)
    validate(config)
    return config


def _deep_merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate(config: ExperimentConfig) -> None:
    if config.protocol.scheme not in {"group_kfold", "loso"}:
        raise ValueError(f"Unknown protocol scheme {config.protocol.scheme!r}")
    if config.input.normalization not in {"window_zscore", "none"}:
        raise ValueError(f"Unknown normalization {config.input.normalization!r}")
    if config.input.alignment not in {"none", "euclidean"}:
        raise ValueError(f"Unknown alignment {config.input.alignment!r}")
    if not 0 < config.protocol.inner_val_fraction < 0.5:
        raise ValueError("inner_val_fraction must be in (0, 0.5)")
    if config.training.monitor not in {"val_loss", "val_balanced_accuracy"}:
        raise ValueError(f"Unknown monitor {config.training.monitor!r}")
    if not config.seeds:
        raise ValueError("At least one seed is required")
