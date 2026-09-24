# stewbench — subject-independent EEG workload benchmark on STEW

Rest vs. SIMKAP-multitasking (low vs. high mental workload) classification on
the STEW dataset (Lim, Sourina & Wang, IEEE TNSRE 2018; 48 subjects, Emotiv
EPOC, 14 channels, 128 Hz), with:

- **25 baselines** on identical subject-independent folds — 8 classical
  (spectral features + sLDA / LR / SVM / RF / GBDT; Riemannian MDM, tangent-space
  LR, re-centred tangent-space LR) and 17 deep (1D-CNN, RNN, LSTM, BiLSTM, GRU,
  CNN-LSTM, EEGNet, ShallowConvNet, DeepConvNet, EEG-TCNet, TSception,
  EEG Conformer, ATCNet, ViT-STFT, Swin-STFT, electrode GCN, DGCNN).
- **Leakage-safe protocol**: whole subjects per fold, subject-disjoint inner
  validation for early stopping/tuning, test subjects touched once.
- **Metrics**: accuracy, balanced accuracy, macro-F1, sensitivity, specificity,
  Cohen's κ, MCC, ROC-AUC, PR-AUC, Brier, log-loss, ECE; window, recording and
  per-subject levels; subject-cluster bootstrap 95% CIs; parameters, FLOPs,
  latency.
- **Statistics**: Friedman + Iman–Davenport, Nemenyi critical-difference
  diagram, pairwise Wilcoxon signed-rank with Holm correction and rank-biserial
  effect sizes.
- **Augmentation study**: 8 standard EEG augmentations vs. CWGAN-GP vs.
  conditional DDPM, trained inside each fold, across training-subject budgets
  (data-efficiency curve), with synthetic-data quality metrics.
- **Neurophysiology check**: task-vs-rest band-power topomaps (paired t, FDR).
- **Interpretability**: channel-permutation and band-removal importance.

## Quick start (NVIDIA workstation, NGC container)

```bash
git clone <this repo> && cd research-eeg
# 1. Put STEW files in data/raw/stew/  (sub01_lo.txt ... sub48_hi.txt, ratings.txt)
# 2. Check your driver with nvidia-smi; default image needs driver >= 570
NGC_TAG=25.01-py3 docker/run.sh build
docker/run.sh check                  # GPU + dataset check
docker/run.sh smoke                  # whole pipeline on a tiny synthetic fixture
docker/run.sh benchmark --config configs/benchmark_quick.yaml   # ~fast sanity run on real data
DETACH=1 docker/run.sh benchmark --config configs/benchmark.yaml  # full paper run (resumable)
docker logs -f stewbench
DETACH=1 docker/run.sh augment --config configs/augmentation.yaml
docker/run.sh neuro --config configs/benchmark.yaml
```

Results land in `outputs/stew/<experiment>/report/` (`REPORT.md`, `tables/*.{csv,tex,md}`,
`figures/*.{pdf,png}`). Interrupted runs resume where they stopped.

Without Docker: install PyTorch, then `pip install -r requirements.txt && pip install -e .`
and use `python -m stewbench <command>`.

## Configs

| File | Purpose |
|---|---|
| `configs/benchmark.yaml` | main benchmark: 25 models × 5 seeds × 10 subject folds |
| `configs/benchmark_quick.yaml` | 1 seed, 5 folds, short training — setup check only |
| `configs/benchmark_loso.yaml` | leave-one-subject-out, for comparison with published STEW numbers |
| `configs/ablation_euclidean_alignment.yaml` | transductive Euclidean Alignment ablation |
| `configs/ablation_window_2s.yaml` | 2 s windows |
| `configs/augmentation.yaml` | augmentation × generator × training-subject budget study |

## Protocol notes

- Band-pass 0.5–45 Hz (zero-phase Butterworth), 4 s windows, 50% overlap,
  class-agnostic per-subject MAD artifact rejection, per-window z-scoring.
- Per-recording normalization is deliberately not offered: each STEW recording
  is one class, so recording statistics would encode the label.
- Methods marked **transductive** (re-centred Riemannian, Euclidean Alignment)
  use unlabelled test-subject data and are reported separately.
- Identical training recipe for all deep models (AdamW, warm-up + cosine,
  early stopping on inner-validation loss); no tuning on test data.
- Kernel lengths of CNNs designed for 250 Hz are rescaled to 128 Hz.

## Status

The code has been exercised end to end on a synthetic fixture only; no STEW
results have been produced yet. Numbers from `smoke` or fixture runs are
labelled as such and must never be reported.

## Tests

```bash
python -m pytest -q tests                 # full suite (slow on CPU)
python -m pytest -q tests/test_splits.py tests/test_metrics_stats.py tests/test_data.py  # fast
```
