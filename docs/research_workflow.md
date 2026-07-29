# Research Workflow

This repo should stay code-first. Raw datasets, processed windows, GAN samples,
model checkpoints, and run outputs are large research artifacts, so they are
ignored by Git and shared through Drive, a GitHub Release, Hugging Face, or an
institutional storage folder.

## Directory Layout

```text
data/
  raw/
    dreamer/      # DREAMER.mat, DREAMER.pdf
    stew/         # original STEW files
    iub/          # original IUB files
    dasps/        # original DASPS files
  processed/
    dreamer/      # subject_*.npz, split.json, synthetic_train_<run>.npz
    stew/
    iub/
    dasps/

models/
  dreamer/<run_name>/cwgan_gp_generator.pt
  dreamer/<run_name>/cwgan_gp_critic.pt
  stew/<run_name>/
  iub/<run_name>/
  dasps/<run_name>/

outputs/
  dreamer/<run_name>/gan_training_loss.png
  dreamer/<run_name>/gan_waveform_check.png
  dreamer/<run_name>/gan_tsne_check.png
  dreamer/baseline_results.json
  dreamer/single_split_results.json

runs/
  <dataset>/<run_name>/manifest.json
  <dataset>/<run_name>/training_history.json
```

## Naming Rule

Use descriptive run names:

```text
gan_400epoch
gan_400epoch_seed42_frac25
eegnet_adapted_30epoch_without_gan
eegnet_adapted_30epoch_with_gan_400epoch
```

Avoid overwriting final artifacts. If you rerun an important experiment, create
a new run name.

## DREAMER Commands

Sanity-check raw DREAMER:

```bash
python -m src.data_loader
```

Preprocess:

```bash
python -m src.preprocessing --dataset dreamer
```

Create the fixed subject-independent split:

```bash
python -m src.split --dataset dreamer
```

Train a GAN:

```bash
python -m src.train_gan --dataset dreamer --run_name gan_400epoch --epochs 400 --batch_size 64
```

Train one classifier without GAN:

```bash
python -m src.train_baseline_single --dataset dreamer --model eegnet_adapted --epochs 30
```

Train one classifier with the saved GAN data:

```bash
python -m src.train_baseline_single --dataset dreamer --model eegnet_adapted --use_gan --gan_run gan_400epoch --epochs 30
```

Run cross-validation baselines:

```bash
python -m src.train_baseline --dataset dreamer --model eegnet_adapted --epochs 30 --folds 5
```

Run the stricter per-fold GAN comparison:

```bash
python -m src.compare_gan_augmentation --dataset dreamer --model eegnet_adapted --gan_epochs 200 --clf_epochs 30 --folds 5 --synth_fraction 0.25
```

Audit the local setup:

```bash
python scripts/check_setup.py --dataset dreamer
```

## STEW Research Protocol

STEW is the active/default dataset. Labels are experimental conditions:
`lo/rest=0` and `hi/SIMKAP multitasking=1`. Describe the task as workload
classification, not validated clinical-stress diagnosis.

### Recommended STEW Commands

Create the subject-independent fixed split once:

```bash
python3 -m src.split --dataset stew
```

Train and evaluate a real-only classifier:

```bash
python3 -m src.train_baseline_single \
  --dataset stew \
  --model gru \
  --epochs 30 \
  --seed 42
```

Train the conditional GAN and generate 25% additional windows for **each**
class:

```bash
python3 -m src.train_gan \
  --dataset stew \
  --run_name stew_gan_seed42_frac25 \
  --epochs 400 \
  --augmentation_fraction 0.25 \
  --seed 42
```

This leaves the real STEW files unchanged. It saves only the generated training
windows to
`data/processed/stew/synthetic_train_stew_gan_seed42_frac25.npz`, together with
the dataset, seed, GAN-training subjects, classifier-validation subjects, and
held-out test subjects.

Train and evaluate the same classifier with those synthetic windows added only
to its inner-training partition:

```bash
python3 -m src.train_baseline_single \
  --dataset stew \
  --model gru \
  --use_gan \
  --gan_run stew_gan_seed42_frac25 \
  --epochs 30 \
  --seed 42
```

The two classifier commands use the same split, initialization seed, real-only
validation set, and real-only test set. Once both finish,
`outputs/stew/single_split_results.json` contains their metrics and the terminal
prints the accuracy and macro-F1 deltas.

For the paper-quality comparison, prefer the stricter paired cross-validation
command. It trains a separate GAN inside every fold, evaluates real-only versus
real-plus-synthetic on identical held-out subjects, and never reuses one saved
synthetic dataset across folds:

```bash
python3 -m src.compare_gan_augmentation \
  --dataset stew \
  --model gru \
  --gan_epochs 400 \
  --clf_epochs 30 \
  --folds 5 \
  --synth_fraction 0.25 \
  --gan_cache_name stew_cv_gan_seed42_frac25 \
  --seed 42 \
  --run_name stew_gru_gan_cv_seed42_frac25
```

An augmentation fraction of `0.25` adds 25% of the real class-0 count and 25%
of the real class-1 count. It increases both conditions proportionally; it is
not a minority-class balancing operation. GAN augmentation is an experimental
factor, not a guaranteed accuracy improvement, so predeclare the fraction and
repeat the comparison across several seeds.

### Active Classifier Suite

The default real-only baseline command runs:

```text
1dcnn, rnn, lstm, bilstm, gru, gnn, vit, swin
```

`lstm` is unidirectional and `bilstm` is bidirectional. Historical `lstm`
artifacts created before this distinction used the bidirectional architecture
and must be rerun for a correctly labeled comparison.

ViT and Swin are compact EEG adaptations, not exact reproductions of the
original image models. Both compute a Hann-window log-magnitude STFT internally
(`n_fft=64`, `hop_length=16`, 0-45 Hz), ensuring that real and GAN-generated raw
EEG receive the identical transformation. The GNN uses the 14 electrodes as
nodes and a symmetric three-nearest-neighbor graph based on the MNE
`standard_1020` three-dimensional electrode coordinates.

EEGNetAdapted, DeepConvNetAdapted, ShallowConvNetAdapted, and TemporalCNN remain
available by explicit `--model` name but are excluded from the default suite.

Run all active models without GAN:

```bash
python3 -m src.train_baseline \
  --dataset stew \
  --epochs 30 \
  --folds 5 \
  --seed 42 \
  --run_name stew_active_real_only_seed42
```

Run the paired per-fold GAN comparison for every active model:

```bash
for model in 1dcnn rnn lstm bilstm gru gnn vit swin; do
  python3 -m src.compare_gan_augmentation \
    --dataset stew \
    --model "$model" \
    --gan_epochs 400 \
    --clf_epochs 30 \
    --folds 5 \
    --synth_fraction 0.25 \
    --gan_cache_name stew_cv_gan_seed42_frac25 \
    --seed 42 \
    --run_name "stew_${model}_gan_cv_seed42_frac25"
done
```

The first model trains and caches one subject-isolated GAN per fold. Later
models validate and reuse those exact fold-specific synthetic samples, avoiding
redundant GAN training while preserving a paired comparison. This creates a
complete real-only versus GAN-augmented result for each architecture. For every
fold, the output includes per-epoch training and real-validation accuracy/loss,
selected-checkpoint training/validation accuracy/loss, one final real holdout
accuracy/loss, macro-F1 and the broader research metrics. Each completed model is merged into
`outputs/stew/gan_comparison_master_table.csv`. Never plot or inspect holdout
performance at every epoch.

### Final Comparison Outputs

Each model run writes:

```text
outputs/stew/<run_name>/comparison_table.csv
outputs/stew/<run_name>/comparison_performance.png
outputs/stew/<run_name>/comparison_classification_diagnostics.png
outputs/stew/<run_name>/fold_<n>/classifier_learning_curves.png
outputs/stew/<run_name>/fold_<n>/synthetic_quality.json
outputs/stew/<run_name>/fold_<n>/real_vs_synthetic_psd.png
outputs/stew/<run_name>/fold_<n>/real_vs_synthetic_channel_correlation.png
outputs/stew/<run_name>.json

runs/stew/<run_name>/comparison_summary.json
runs/stew/<run_name>/fold_<n>/manifest.json
runs/stew/<run_name>/fold_<n>/gan_training_history.json

models/stew/<run_name>/fold_<n>/<model>_without_gan.pt
models/stew/<run_name>/fold_<n>/<model>_with_gan.pt
```

The shared fold GAN cache additionally writes its synthetic arrays, generator
and critic checkpoints, GAN loss plots, and protocol manifests. The master
`outputs/stew/gan_comparison_master_table.csv` combines completed model runs.
Its before/after columns include selected-checkpoint training accuracy/loss,
real-validation accuracy/loss, final real test accuracy/loss, macro-F1 and
subject-level statistics.

Preprocessing uses 0.5-45 Hz filtering, 4-second windows, 50% overlap, a
STEW-calibrated MAD artifact multiplier of 60, and per-window/per-channel
z-scoring. The multiplier was selected using fixed training subjects only
(4.73% rejection; low=3.06%, high=6.40%); test subjects were excluded from
calibration. Window-local normalization prevents one held-out recording from
affecting another evaluated window.

The fixed experiment has three subject-disjoint pools:

```text
inner-training subjects -> GAN training and classifier gradient updates
inner-validation subjects -> classifier checkpoint selection only
test subjects -> final metric only
```

Synthetic `.npz` files store these subject lists and the random seed.
`train_baseline_single` rejects legacy or mismatched synthetic files. The
per-fold comparison follows the same isolation independently inside every fold.

GAN runs save a shareable manifest and history under `runs/`, plus a quantitative
quality report under `outputs/`. Run multiple seeds and report mean and standard
deviation; t-SNE alone is not evidence of synthetic EEG validity.

CWGAN-GP uses the reference optimizer settings `Adam(lr=1e-4, betas=(0, 0.9))`,
five critic updates, and gradient-penalty coefficient 10. Augmentation is a
predeclared percentage of each real class, while quality evaluation always
generates an independent balanced sample from both classes.

Report both window-level and subject-condition-level metrics: accuracy,
balanced accuracy, macro precision/recall/F1, sensitivity, specificity, MCC,
ROC-AUC, average precision, per-class metrics, confusion matrices, model size,
training time, and inference time. Confidence intervals resample whole subjects
so overlapping windows are never treated as independent bootstrap units.

Paired real-only versus augmented comparisons use the same folds and classifier
seeds. The primary subject-condition analysis pools out-of-fold predictions from
all 48 held-out subjects and reports a subject-cluster bootstrap interval plus a
subject-wise randomization p-value. Holm correction is applied across reported
paired metrics. The fold-level exact sign-flip test is retained as a secondary
diagnostic; with five folds its smallest possible two-sided p-value is 0.0625.

Use `--include_simple_augmentation` to compare CWGAN-GP with conventional
Gaussian-noise, time-shift, and channel-dropout augmentation at the same sample
fraction. Predeclare fractions and seeds; never select either from final
held-out performance.

Synthetic quality must be reported across fidelity (PSD/band power,
autocorrelation, covariance, MMD), diversity (density and coverage), utility
(TRTR/TSTR and real-only versus real+synthetic classifiers), and memorization
(synthetic-to-training versus synthetic-to-unseen nearest-neighbor distances).
Nearest-neighbor results are diagnostics and must not be described as formal
privacy guarantees.

Every baseline and GAN-comparison fold saves checkpoints, selected epoch,
training history, subject lists, raw/split checksums, Git state, package versions,
and preprocessing settings under `runs/`. Commit the code before a final run so
the manifest records `dirty=false`.

## Adding IUB and DASPS

IUB and DASPS are planned and intentionally rejected by training CLIs until
implemented. For each new dataset, add a `DatasetSpec` and loader that converts raw files into the same
standard processed format as DREAMER:

```text
subject_XX.npz
  windows:   float32, shape (N, T, C)
  trial_idx: int array, shape (N,)
  label fields needed by that dataset labeler
```

Keep the split rule subject-independent unless the dataset has no reliable
subject IDs. Never train the GAN on test subjects.

### STEW Notes

Raw STEW files should live at:

```text
data/raw/stew/stew_dataset/
```

The dataset has 48 subjects, two conditions per subject, 14 Emotiv EPOC EEG
channels, 128 Hz sampling, and 2.5 minutes per condition.

File convention:

```text
sub01_lo.txt  # subject 1 at rest / low workload
sub01_hi.txt  # subject 1 during SIMKAP multitasking / high workload
ratings.txt  # subject number, rest rating, test rating
```

Column order:

```text
AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
```

Ratings for subjects 5, 24, and 42 are unavailable according to the IEEE
DataPort page. Decide whether to exclude those subjects for rating-based labels
or keep them for condition-based labels (`lo=0`, `hi=1`).

Current code uses condition-based labels:

```text
lo/rest = 0
hi/multitasking = 1
```

This makes all 48 subjects usable, including subjects 5, 24, and 42 whose
ratings are unavailable. A rating-based label mode can be added later if needed.

## Sharing With The Team

Send code through Git. Send artifacts separately:

```text
data/processed/<dataset>/synthetic_train_<run>.npz
models/<dataset>/<run>/
outputs/<dataset>/<run>/
runs/<dataset>/<run>/manifest.json
```

For a paper, record dataset, preprocessing config, label rule, split seed, GAN
epochs, classifier epochs, and model name for every reported result.
