"""
Trains the CWGAN-GP on REAL TRAINING DATA ONLY from one CV fold (never on
validation/test subjects -- that would leak information). Produces:
    - trained generator checkpoint (models/<dataset>/<run>/cwgan_gp_generator.pt)
    - loss curve plot (outputs/<dataset>/<run>/gan_training_loss.png)
    - real vs synthetic waveform comparison plot (outputs/<dataset>/<run>/gan_waveform_check.png)
    - real vs synthetic t-SNE overlay (outputs/<dataset>/<run>/gan_tsne_check.png)

The fixed-split pipeline excludes both classifier-validation and test subjects
from GAN training. The cross-validation comparison trains a separate GAN on
each fold's inner-training subjects.

Usage:
    python -m src.train_gan --dataset stew --run_name gan_400epoch_seed42 --epochs 400 --seed 42
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")  # no display needed, just save figures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from . import config
from .datasets import EEGWindowDataset
from .gan import Generator, Critic, gradient_penalty, LATENT_DIM, weights_init
from .labeling import build_dataset
from .preprocessing import load_processed
from .provenance import build_provenance, write_json
from .reproducibility import set_seed
from .synthetic_quality import evaluate_synthetic_quality


N_CRITIC = 5  # critic updates per generator update (standard WGAN-GP ratio)
LAMBDA_GP = 10.0
GAN_LEARNING_RATE = 1e-4
ADAM_BETAS = (0.0, 0.9)
DEFAULT_AUGMENTATION_FRACTION = 0.25
QUALITY_SAMPLES_PER_CLASS = 256


import time

def train_gan(X_train, y_train, device, epochs=200, batch_size=64,
              lr: float = GAN_LEARNING_RATE, betas: tuple[float, float] = ADAM_BETAS,
              seed: int = config.RANDOM_SEED):
    set_seed(seed)
    n_timepoints, n_channels = X_train.shape[1], X_train.shape[2]
    gen = Generator(n_channels=n_channels, n_timepoints=n_timepoints).to(device)
    crit = Critic(n_channels=n_channels, n_timepoints=n_timepoints).to(device)
    gen.apply(weights_init)
    crit.apply(weights_init)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_crit = optim.Adam(crit.parameters(), lr=lr, betas=betas)

    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        EEGWindowDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    if len(loader) == 0:
        raise ValueError(
            f"GAN training has {len(X_train)} samples, fewer than batch_size={batch_size}."
        )

    history = {"critic_loss": [], "gen_loss": [], "wasserstein_estimate": []}

    epoch_times = []
    for epoch in range(epochs):
        t0 = time.time()
        epoch_critic_loss, epoch_gen_loss, epoch_wdist = [], [], []

        data_iterator = iter(loader)
        for _ in range(len(loader)):
            # --- Train critic ---
            for _ in range(N_CRITIC):
                try:
                    real, labels = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(loader)
                    real, labels = next(data_iterator)
                real, labels = real.to(device), labels.to(device)
                b = real.size(0)
                z = torch.randn(b, LATENT_DIM, device=device)
                fake = gen(z, labels).detach()

                critic_real = crit(real, labels)
                critic_fake = crit(fake, labels)
                gp = gradient_penalty(crit, real, fake, labels, device)

                critic_loss = -(critic_real.mean() - critic_fake.mean()) + LAMBDA_GP * gp

                opt_crit.zero_grad()
                critic_loss.backward()
                opt_crit.step()
                epoch_critic_loss.append(critic_loss.item())
                epoch_wdist.append((critic_real.mean() - critic_fake.mean()).item())

            # --- Train generator ---
            z = torch.randn(b, LATENT_DIM, device=device)
            fake = gen(z, labels)
            gen_loss = -crit(fake, labels).mean()

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            epoch_gen_loss.append(gen_loss.item())

        history["critic_loss"].append(float(np.mean(epoch_critic_loss)))
        history["gen_loss"].append(float(np.mean(epoch_gen_loss)))
        history["wasserstein_estimate"].append(float(np.mean(epoch_wdist)))

        epoch_times.append(time.time() - t0)
        avg_epoch_time = np.mean(epoch_times[-5:])  # rolling average, last 5 epochs
        remaining = epochs - (epoch + 1)
        eta_seconds = remaining * avg_epoch_time

        if (epoch + 1) <= 3 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  critic_loss={history['critic_loss'][-1]:.3f}  "
                  f"gen_loss={history['gen_loss'][-1]:.3f}  "
                  f"wasserstein_est={history['wasserstein_estimate'][-1]:.3f}  "
                  f"[{avg_epoch_time:.1f}s/epoch, ETA {eta_seconds/60:.1f} min]")

    return gen, crit, history


def generate_synthetic(gen, n_samples_by_class: dict, device, n_timepoints=512,
                       n_channels=14, generation_batch_size: int = 512):
    """
    Generates synthetic windows per class.
    n_samples_by_class: e.g. {0: 500, 1: 500} or {1: 800} to generate only class 1.
    Returns X_synth, y_synth.
    """
    gen.eval()
    X_list, y_list = [], []
    with torch.no_grad():
        for cls, n in n_samples_by_class.items():
            if n <= 0:
                continue
            for start in range(0, n, generation_batch_size):
                current_batch = min(generation_batch_size, n - start)
                z = torch.randn(current_batch, LATENT_DIM, device=device)
                labels = torch.full(
                    (current_batch,), cls, dtype=torch.long, device=device
                )
                X_list.append(gen(z, labels).cpu().numpy().astype(np.float32))
                y_list.append(np.full(current_batch, cls, dtype=np.int64))
    gen.train()
    if not X_list:
        return (np.empty((0, n_timepoints, n_channels), dtype=np.float32),
                np.empty((0,), dtype=np.int64))
    X_synth = np.concatenate(X_list, axis=0)
    y_synth = np.concatenate(y_list, axis=0)
    return X_synth, y_synth


# ---------------------------------------------------------------------------
# Validation plots
# ---------------------------------------------------------------------------
def plot_loss_curve(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["critic_loss"], label="critic loss")
    axes[0].plot(history["gen_loss"], label="generator loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("CWGAN-GP training losses")

    axes[1].plot(history["wasserstein_estimate"], color="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Wasserstein estimate")
    axes[1].set_title("Wasserstein distance estimate (should stabilize, not diverge)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved loss curve to {out_path}")


def plot_waveform_comparison(X_real, y_real, X_synth, y_synth, out_path,
                             channel_idx=0, class_names=("class 0", "class 1")):
    available_classes = [c for c in (0, 1) if (y_synth == c).any()]
    if not available_classes:
        print(f"  [skip] no synthetic samples for any class -- skipping waveform plot")
        return

    fig, axes = plt.subplots(len(available_classes), 2, figsize=(12, 3 * len(available_classes)),
                               squeeze=False)
    for row, cls in enumerate(available_classes):
        real_sample = X_real[y_real == cls][0][:, channel_idx]
        synth_sample = X_synth[y_synth == cls][0][:, channel_idx]
        axes[row, 0].plot(real_sample)
        axes[row, 0].set_title(f"REAL: {class_names[cls]}, channel={channel_idx}")
        axes[row, 1].plot(synth_sample, color="orange")
        axes[row, 1].set_title(f"SYNTHETIC: {class_names[cls]}, channel={channel_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved waveform comparison to {out_path}")


def plot_tsne_overlay(X_real, y_real, X_synth, y_synth, out_path, max_points=1000,
                      class_names=("class 0", "class 1"),
                      seed: int = config.RANDOM_SEED):
    from sklearn.manifold import TSNE

    if len(X_synth) < 2:
        print("  [skip] fewer than two synthetic samples -- skipping t-SNE plot")
        return

    # Flatten windows to feature vectors for t-SNE (simple approach: mean+std per channel)
    def summarize(X):
        return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)  # (N, 2*C)

    # Subsample for speed
    def subsample(X, y, n):
        if len(X) <= n:
            return X, y
        idx = np.random.choice(len(X), n, replace=False)
        return X[idx], y[idx]

    X_real_s, y_real_s = subsample(X_real, y_real, max_points)
    X_synth_s, y_synth_s = subsample(X_synth, y_synth, max_points)

    feat_real = summarize(X_real_s)
    feat_synth = summarize(X_synth_s)

    combined = np.concatenate([feat_real, feat_synth], axis=0)
    labels_combined = np.concatenate([
        np.array(["real"] * len(feat_real)),
        np.array(["synthetic"] * len(feat_synth)),
    ])
    class_combined = np.concatenate([y_real_s, y_synth_s])

    perplexity = min(30, len(combined) - 1)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    embedded = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 6))
    for source, marker in [("real", "o"), ("synthetic", "x")]:
        for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
            mask = (labels_combined == source) & (class_combined == cls)
            ax.scatter(embedded[mask, 0], embedded[mask, 1], marker=marker, color=color,
                       alpha=0.5, label=f"{source}: {class_names[cls]}", s=15)
    ax.legend()
    ax.set_title("t-SNE: real vs synthetic EEG (o=real, x=synthetic)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved t-SNE overlay to {out_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def train_gan_pipeline(epochs=200, batch_size=64, n_synth_per_class=None,
                       dataset: str = config.DEFAULT_DATASET, run_name: str | None = None,
                       seed: int = config.RANDOM_SEED, overwrite: bool = False,
                       augmentation_fraction: float = DEFAULT_AUGMENTATION_FRACTION,
                       quality_samples_per_class: int = QUALITY_SAMPLES_PER_CLASS):
    """
    Trains CWGAN-GP on inner-training subjects only. Classifier-validation and
    test subjects are excluded. Synthetic data includes protocol metadata so
    train_baseline_single.py can reject mismatched runs.
    """
    from .split import apply_split_with_groups, inner_group_split_indices, load_split

    dataset = config.normalize_dataset_name(dataset)
    if augmentation_fraction < 0:
        raise ValueError("augmentation_fraction must be non-negative.")
    if quality_samples_per_class <= 0:
        raise ValueError("quality_samples_per_class must be positive.")
    run_name = run_name or f"gan_{epochs}epoch"
    processed_dir = config.processed_dir(dataset)
    model_dir = config.model_dir(dataset, run_name)
    output_dir = config.output_dir(dataset, run_name)
    run_dir = config.run_dir(dataset, run_name)
    synth_path = processed_dir / f"synthetic_train_{run_name}.npz"
    existing_artifacts = [path for path in (model_dir, output_dir, run_dir, synth_path) if path.exists()]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            "Run artifacts already exist; choose a new --run_name or pass --overwrite: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    device = config.get_device()
    set_seed(seed)
    print(f"Using device: {device}")
    print(f"Dataset: {dataset} | run: {run_name}")

    processed = load_processed(dataset=dataset)
    X, y, groups = build_dataset(processed)

    split_info = load_split(dataset=dataset)
    X_train, y_train, groups_train, X_test, y_test, groups_test = apply_split_with_groups(
        X, y, groups, split_info
    )
    inner_train_idx, inner_val_idx = inner_group_split_indices(groups_train, seed=seed)
    if inner_val_idx is None:
        raise ValueError("At least three training subjects are required for strict GAN isolation.")
    X_gan, y_gan = X_train[inner_train_idx], y_train[inner_train_idx]
    gan_train_subjects = sorted(np.unique(groups_train[inner_train_idx]).astype(int).tolist())
    inner_val_subjects = sorted(np.unique(groups_train[inner_val_idx]).astype(int).tolist())
    class0_name, class1_name = config.class_names(dataset)
    print(f"GAN inner-train: X={X_gan.shape}, subjects={gan_train_subjects}")
    print(f"Excluded classifier-validation subjects: {inner_val_subjects}")
    print(f"Held-out test subjects: {split_info['test_subjects']}")
    print(f"GAN class balance: class0={int((y_gan == 0).sum())} ({class0_name}), "
          f"class1={int((y_gan == 1).sum())} ({class1_name})")

    gen, crit, history = train_gan(
        X_gan,
        y_gan,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), model_dir / "cwgan_gp_generator.pt")
    torch.save(crit.state_dict(), model_dir / "cwgan_gp_critic.pt")
    print(f"Saved generator/critic checkpoints to {model_dir}")

    # Validation plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, output_dir / "gan_training_loss.png")

    # Generate a predeclared fraction for both classes. This tests general
    # augmentation rather than conflating augmentation with class balancing.
    n_class0 = int((y_gan == 0).sum())
    n_class1 = int((y_gan == 1).sum())
    if n_synth_per_class is None:
        n_synth_per_class = {
            0: int(round(n_class0 * augmentation_fraction)),
            1: int(round(n_class1 * augmentation_fraction)),
        }
        print(
            f"Augmentation fraction={augmentation_fraction:.2f}: generating "
            f"class0={n_synth_per_class[0]}, class1={n_synth_per_class[1]}."
        )
    else:
        n_synth_per_class = {0: n_synth_per_class, 1: n_synth_per_class}

    X_synth_list, y_synth_list = [], []
    for cls, n in n_synth_per_class.items():
        if n <= 0:
            continue
        X_c, y_c = generate_synthetic(gen, {cls: n}, device, n_timepoints=X.shape[1], n_channels=X.shape[2])
        X_synth_list.append(X_c)
        y_synth_list.append(y_c)

    X_synth = np.concatenate(X_synth_list, axis=0) if X_synth_list else np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32)
    y_synth = np.concatenate(y_synth_list, axis=0) if y_synth_list else np.empty((0,), dtype=np.int64)

    # SAVE the synthetic data to disk -- this is the actual deliverable file
    np.savez_compressed(
        synth_path,
        X_synth=X_synth,
        y_synth=y_synth,
        dataset=np.array(dataset),
        random_seed=np.array(seed, dtype=np.int64),
        gan_train_subjects=np.asarray(gan_train_subjects, dtype=np.int64),
        inner_val_subjects=np.asarray(inner_val_subjects, dtype=np.int64),
        test_subjects=np.asarray(split_info["test_subjects"], dtype=np.int64),
    )
    print(f"\nSaved {len(y_synth)} synthetic windows to {synth_path}")
    print(f"  synthetic class 0 ({class0_name}): {(y_synth==0).sum()}")
    print(f"  synthetic class 1 ({class1_name}): {(y_synth==1).sum()}")

    # Quality evaluation is always balanced across both labels and is separate
    # from the samples used to augment classifier training.
    X_quality, y_quality = generate_synthetic(
        gen,
        {0: quality_samples_per_class, 1: quality_samples_per_class},
        device,
        n_timepoints=X.shape[1],
        n_channels=X.shape[2],
    )

    # Validation plots comparing real GAN-training data to balanced quality samples
    class_names = (class0_name, class1_name)
    plot_waveform_comparison(
        X_gan, y_gan, X_quality, y_quality,
        output_dir / "gan_waveform_check.png", class_names=class_names,
    )
    plot_tsne_overlay(
        X_gan, y_gan, X_quality, y_quality,
        output_dir / "gan_tsne_check.png", class_names=class_names, seed=seed,
    )

    quality = evaluate_synthetic_quality(X_gan, y_gan, X_quality, y_quality)
    quality_path = output_dir / "synthetic_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality, f, indent=2)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "training_history.json", history)
    split_path = config.processed_dir(dataset) / "split.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "run_name": run_name,
        "model": "CWGAN-GP",
        "random_seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": GAN_LEARNING_RATE,
        "adam_betas": list(ADAM_BETAS),
        "n_critic": N_CRITIC,
        "lambda_gp": LAMBDA_GP,
        "augmentation_fraction": augmentation_fraction,
        "quality_samples_per_class": quality_samples_per_class,
        "window_samples": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
        "normalization": config.STEW_NORMALIZATION if dataset == "stew" else config.NORMALIZATION,
        "artifact_mad_multiplier": config.artifact_rejection_mad_multiplier(dataset),
        "gan_train_subjects": gan_train_subjects,
        "inner_val_subjects": inner_val_subjects,
        "test_subjects": split_info["test_subjects"],
        "real_gan_train_class_counts": {"0": n_class0, "1": n_class1},
        "synthetic_class_counts": {
            "0": int((y_synth == 0).sum()),
            "1": int((y_synth == 1).sum()),
        },
        "artifacts": {
            "generator": str((model_dir / "cwgan_gp_generator.pt").relative_to(config.PROJECT_ROOT)),
            "critic": str((model_dir / "cwgan_gp_critic.pt").relative_to(config.PROJECT_ROOT)),
            "synthetic_data": str(synth_path.relative_to(config.PROJECT_ROOT)),
            "quality_report": str(quality_path.relative_to(config.PROJECT_ROOT)),
        },
        "provenance": build_provenance(dataset, split_path),
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Saved reproducibility manifest and history to {run_dir}")

    print(f"\nGAN training complete. Review:")
    print(f"  {output_dir / 'gan_training_loss.png'}")
    print(f"  {output_dir / 'gan_waveform_check.png'}")
    print(f"  {output_dir / 'gan_tsne_check.png'}")
    print(f"before trusting {synth_path} for the with-GAN classifier run.")

    return gen, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--augmentation_fraction", type=float, default=DEFAULT_AUGMENTATION_FRACTION)
    parser.add_argument("--quality_samples_per_class", type=int, default=QUALITY_SAMPLES_PER_CLASS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_gan_pipeline(epochs=args.epochs, batch_size=args.batch_size,
                       dataset=args.dataset, run_name=args.run_name, seed=args.seed,
                       overwrite=args.overwrite,
                       augmentation_fraction=args.augmentation_fraction,
                       quality_samples_per_class=args.quality_samples_per_class)
