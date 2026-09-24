"""Uniform training loop used for every deep model.

Identical budget and optimisation recipe for all architectures (AdamW,
warm-up + cosine schedule, early stopping on subject-disjoint inner
validation). No per-model tuning on test data is possible: the test
subjects are never passed to this module.
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import build_deep
from .settings import TrainingConfig
from .utils import set_seed

warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")

# augment(x, y, generator) -> (x_aug, target) where target is either integer
# labels (B,) or class-probability targets (B, n_classes) (e.g. mixup).
OnlineAugment = Callable[[torch.Tensor, torch.Tensor, torch.Generator], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    epochs_run: int
    train_seconds: float
    history: dict = field(default_factory=dict)


def _autocast(device: torch.device, enabled: bool):
    if not (enabled and device.type == "cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _lr_lambda(total_steps: int, warmup_steps: int, scheduler: str):
    def fn(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if scheduler == "none":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray | torch.Tensor, device, batch_size: int = 512,
                  amp: bool = False) -> np.ndarray:
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    outputs = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(xb)
        outputs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(outputs).numpy() if outputs else np.empty((0, 2), dtype=np.float32)


def _evaluate(model, X, y, device, amp) -> dict:
    probabilities = predict_proba(model, X, device, amp=amp)
    y_np = y.cpu().numpy() if torch.is_tensor(y) else y
    loss = float(F.nll_loss(torch.log(torch.as_tensor(probabilities).clamp_min(1e-7)),
                            torch.as_tensor(y_np)).item())
    pred = probabilities.argmax(1)
    recalls = [np.mean(pred[y_np == c] == c) for c in np.unique(y_np)]
    return {"loss": loss, "accuracy": float(np.mean(pred == y_np)), "balanced_accuracy": float(np.mean(recalls))}


def train_deep(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray, config: TrainingConfig, seed: int,
               device: torch.device, augment: OnlineAugment | None = None,
               X_extra: np.ndarray | None = None, y_extra: np.ndarray | None = None,
               sfreq: int = 128, verbose: bool = False) -> TrainResult:
    """Train one model. ``X_extra``/``y_extra`` (e.g. synthetic windows) are
    appended to the training set only; validation stays real-only."""
    set_seed(seed, config.deterministic)
    if X_extra is not None and len(X_extra):
        X_train = np.concatenate([X_train, X_extra]).astype(np.float32)
        y_train = np.concatenate([y_train, y_extra]).astype(np.int64)
    n_channels, n_times = X_train.shape[1:]
    model = build_deep(model_name, n_channels=n_channels, n_times=n_times, n_classes=2, sfreq=sfreq).to(device)

    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long)

    counts = torch.bincount(yt, minlength=2).float()
    class_weight = (counts.sum() / (2 * counts.clamp_min(1))).to(device)

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")
    steps_per_epoch = max(1, math.ceil(len(Xt) / config.batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(
        config.max_epochs * steps_per_epoch, config.warmup_epochs * steps_per_epoch, config.scheduler))
    use_scaler = config.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Augmentations draw their randomness on CPU (portable across CUDA/MPS/CPU)
    # and move the small random tensors to the batch device.
    aug_generator = torch.Generator(device="cpu").manual_seed(seed + 1)

    minimize = config.monitor == "val_loss"
    best_score, best_state, best_epoch, stale = math.inf if minimize else -math.inf, None, 0, 0
    history = {k: [] for k in ("train_loss", "train_accuracy", "val_loss", "val_accuracy",
                               "val_balanced_accuracy", "lr", "epoch_seconds")}
    started = time.perf_counter()
    epoch = 0
    for epoch in range(1, config.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(len(Xt), generator=generator).to(device)
        loss_sum, correct, seen = 0.0, 0, 0
        for start in range(0, len(order), config.batch_size):
            index = order[start:start + config.batch_size]
            if len(index) < 2:  # BatchNorm needs >1 sample
                continue
            xb, yb = Xt[index], yt[index]
            target = yb
            if augment is not None:
                xb, target = augment(xb, yb, aug_generator)
            with _autocast(device, config.amp):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), target, weight=class_weight,
                                       label_smoothing=config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if config.grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(index)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(index)

        val = _evaluate(model, X_val, yv, device, config.amp)
        history["train_loss"].append(loss_sum / max(1, seen))
        history["train_accuracy"].append(correct / max(1, seen))
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_balanced_accuracy"].append(val["balanced_accuracy"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_seconds"].append(time.perf_counter() - t0)

        score = val["loss"] if minimize else val["balanced_accuracy"]
        improved = score < best_score - 1e-6 if minimize else score > best_score + 1e-6
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if verbose:
            print(f"    epoch {epoch:3d} train_loss={history['train_loss'][-1]:.4f} "
                  f"val_loss={val['loss']:.4f} val_bacc={val['balanced_accuracy']:.3f}")
        if stale >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return TrainResult(model=model, best_epoch=best_epoch, epochs_run=epoch,
                       train_seconds=time.perf_counter() - started, history=history)
