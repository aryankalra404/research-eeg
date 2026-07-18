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

Report both window-level and subject-condition-level metrics. Confidence
intervals resample whole subjects so overlapping windows are never treated as
independent bootstrap units. Paired real-only versus augmented comparisons use
the same folds and classifier seeds and include an exact sign-flip test.

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
