"""
Trains the CWGAN-GP on REAL TRAINING DATA ONLY from one CV fold (never on
validation/test subjects -- that would leak information). Produces:
    - trained generator checkpoint (models/<dataset>/<run>/cwgan_gp_generator.pt)
    - loss curve plot (outputs/<dataset>/<run>/gan_training_loss.png)
    - real vs synthetic waveform comparison plot (outputs/<dataset>/<run>/gan_waveform_check.png)
    - real vs synthetic t-SNE overlay (outputs/<dataset>/<run>/gan_tsne_check.png)

IMPORTANT: run this per-fold if you want strict fold isolation (train GAN
only on that fold's training subjects), or once on the full training pool
if you're doing a simpler single train/test split for the with/without-GAN
comparison your teacher asked for. See train_gan_pipeline() args.

Usage:
    python -m src.train_gan --dataset dreamer --epochs 200
"""

import argparse
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


N_CRITIC = 5  # critic updates per generator update (standard WGAN-GP ratio)
LAMBDA_GP = 10.0


import time

def train_gan(X_train, y_train, device, epochs=200, batch_size=64, lr=1e-4):
    n_timepoints, n_channels = X_train.shape[1], X_train.shape[2]
    gen = Generator(n_channels=n_channels, n_timepoints=n_timepoints).to(device)
    crit = Critic(n_channels=n_channels, n_timepoints=n_timepoints).to(device)
    gen.apply(weights_init)
    crit.apply(weights_init)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_crit = optim.Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))

    loader = DataLoader(EEGWindowDataset(X_train, y_train), batch_size=batch_size,
                         shuffle=True, drop_last=True)

    history = {"critic_loss": [], "gen_loss": [], "wasserstein_estimate": []}

    epoch_times = []
    for epoch in range(epochs):
        t0 = time.time()
        epoch_critic_loss, epoch_gen_loss, epoch_wdist = [], [], []

        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            b = real.size(0)

            # --- Train critic ---
            for _ in range(N_CRITIC):
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

        history["critic_loss"].append(np.mean(epoch_critic_loss))
        history["gen_loss"].append(np.mean(epoch_gen_loss))
        history["wasserstein_estimate"].append(np.mean(epoch_wdist))

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


def generate_synthetic(gen, n_samples_by_class: dict, device, n_timepoints=512, n_channels=14):
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
            z = torch.randn(n, LATENT_DIM, device=device)
            labels = torch.full((n,), cls, dtype=torch.long, device=device)
            fake = gen(z, labels).cpu().numpy()
            X_list.append(fake)
            y_list.append(np.full(n, cls, dtype=np.int64))
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


def plot_waveform_comparison(X_real, y_real, X_synth, y_synth, out_path, channel_idx=0):
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
        axes[row, 0].set_title(f"REAL window, class={cls}, channel={channel_idx}")
        axes[row, 1].plot(synth_sample, color="orange")
        axes[row, 1].set_title(f"SYNTHETIC window, class={cls}, channel={channel_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved waveform comparison to {out_path}")


def plot_tsne_overlay(X_real, y_real, X_synth, y_synth, out_path, max_points=1000):
    from sklearn.manifold import TSNE

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

    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=30)
    embedded = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 6))
    for source, marker in [("real", "o"), ("synthetic", "x")]:
        for cls, color in [(0, "tab:blue"), (1, "tab:red")]:
            mask = (labels_combined == source) & (class_combined == cls)
            ax.scatter(embedded[mask, 0], embedded[mask, 1], marker=marker, color=color,
                       alpha=0.5, label=f"{source} class={cls}", s=15)
    ax.legend()
    ax.set_title("t-SNE: real vs synthetic (o=real, x=synthetic; blue=non-stress, red=stress)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved t-SNE overlay to {out_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def train_gan_pipeline(epochs=200, batch_size=64, n_synth_per_class=None,
                       dataset: str = config.DEFAULT_DATASET, run_name: str | None = None):
    """
    Trains CWGAN-GP on the TRAINING SPLIT ONLY (from src/split.py -- never
    touches test subjects), then generates and SAVES synthetic data to
    data/processed/synthetic_train.npz so it can be inspected, reused, and
    combined with real data by train_baseline_single.py without retraining
    the GAN every time.
    """
    from .split import load_split, apply_split

    dataset = config.normalize_dataset_name(dataset)
    run_name = run_name or f"gan_{epochs}epoch"
    processed_dir = config.processed_dir(dataset)
    model_dir = config.model_dir(dataset, run_name)
    output_dir = config.output_dir(dataset, run_name)
    device = config.get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset} | run: {run_name}")

    processed = load_processed(dataset=dataset)
    X, y, groups = build_dataset(processed)

    split_info = load_split(dataset=dataset)
    X_train, y_train, X_test, y_test = apply_split(X, y, groups, split_info)
    print(f"Training GAN on TRAIN split only: X_train={X_train.shape} "
          f"(test split held out: {X_test.shape[0]} windows, never seen by GAN)")
    print(f"Train class balance: stress={y_train.sum()} ({100*y_train.mean():.1f}%), "
          f"non-stress={len(y_train)-y_train.sum()}")

    gen, crit, history = train_gan(X_train, y_train, device=device, epochs=epochs, batch_size=batch_size)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), model_dir / "cwgan_gp_generator.pt")
    torch.save(crit.state_dict(), model_dir / "cwgan_gp_critic.pt")
    print(f"Saved generator/critic checkpoints to {model_dir}")

    # Validation plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, output_dir / "gan_training_loss.png")

    # Decide how much synthetic data to generate: by default, balance the
    # minority (stress) class up to match the majority (non-stress) class
    # within the TRAIN split only.
    n_class0 = int((y_train == 0).sum())
    n_class1 = int((y_train == 1).sum())
    if n_synth_per_class is None:
        minority_deficit = abs(n_class0 - n_class1)
        n_synth_per_class = {
            0: minority_deficit if n_class0 < n_class1 else 0,
            1: minority_deficit if n_class1 < n_class0 else 0,
        }
        # also generate a modest amount for the majority class so both classes
        # get *some* augmentation, not just the minority -- helps compare
        # "pure rebalancing" vs "general augmentation" if you want that later
        print(f"Auto-balancing: generating {minority_deficit} extra synthetic samples "
              f"for the minority class to match the majority class count.")
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
    synth_path = processed_dir / f"synthetic_train_{run_name}.npz"
    np.savez_compressed(synth_path, X_synth=X_synth, y_synth=y_synth)
    print(f"\nSaved {len(y_synth)} synthetic windows to {synth_path}")
    print(f"  synthetic class 0 (non-stress): {(y_synth==0).sum()}")
    print(f"  synthetic class 1 (stress):     {(y_synth==1).sum()}")

    # Validation plots comparing real TRAIN data to the saved synthetic data
    plot_waveform_comparison(X_train, y_train, X_synth, y_synth, output_dir / "gan_waveform_check.png")
    plot_tsne_overlay(X_train, y_train, X_synth, y_synth, output_dir / "gan_tsne_check.png")

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
    args = parser.parse_args()

    train_gan_pipeline(epochs=args.epochs, batch_size=args.batch_size,
                       dataset=args.dataset, run_name=args.run_name)
