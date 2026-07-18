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
  dreamer/<run_name>/manifest.json
```

## Naming Rule

Use descriptive run names:

```text
gan_400epoch
gan_400epoch_seed42_balanced
eegnet_30epoch_without_gan
eegnet_30epoch_with_gan_400epoch
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
python -m src.train_baseline_single --dataset dreamer --model eegnet --epochs 30
```

Train one classifier with the saved GAN data:

```bash
python -m src.train_baseline_single --dataset dreamer --model eegnet --use_gan --gan_run gan_400epoch --epochs 30
```

Run cross-validation baselines:

```bash
python -m src.train_baseline --dataset dreamer --model eegnet --epochs 30 --folds 5
```

Run the stricter per-fold GAN comparison:

```bash
python -m src.compare_gan_augmentation --dataset dreamer --model eegnet --gan_epochs 200 --clf_epochs 30 --folds 5
```

Audit the local setup:

```bash
python scripts/check_setup.py --dataset dreamer
```

## Adding STEW, IUB, and DASPS

For each new dataset, create a loader that converts raw files into the same
standard processed format as DREAMER:

```text
subject_XX.npz
  windows:   float32, shape (N, T, C)
  trial_idx: int array, shape (N,)
  label fields needed by that dataset labeler
```

Keep the split rule subject-independent unless the dataset has no reliable
subject IDs. Never train the GAN on test subjects.

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
