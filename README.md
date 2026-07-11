# Stress Detection from EEG (Emotiv EPOC) — Research Pipeline

Pipeline: DREAMER preprocessing → CWGAN-GP augmentation → benchmark against
1D-CNN, Vanilla LSTM, EEGNet, DeepConvNet, ShallowConvNet, TemporalCNN.

## Folder structure and what goes where

```
stress-eeg-project/
├── README.md              ← you are here
├── requirements.txt        ← Python dependencies, install with:
│                              pip install -r requirements.txt
│
├── data/
│   ├── raw/                ← RAW, UNTOUCHED source files go here.
│   │                          Put DREAMER.mat here (download from
│   │                          https://zenodo.org/records/546113).
│   │                          Later: STEW/DASPS raw files also go here,
│   │                          each in its own subfolder e.g. raw/stew/, raw/dasps/.
│   │                          Never edit files in raw/ — treat as read-only source of truth.
│   │
│   └── processed/          ← OUTPUT of the preprocessing pipeline (Phase 1).
│                              Filtered, epoched, normalized EEG windows +
│                              labels, saved as .npy/.npz/.pkl.
│                              This is what your training scripts actually load —
│                              never re-run raw preprocessing inside a training loop.
│                              Safe to delete and regenerate anytime from raw/.
│
├── src/                    ← ALL pipeline code (importable Python modules, not
│   │                          one-off scripts). This is the actual pipeline logic.
│   ├── __init__.py
│   ├── config.py            ← single source of truth for every pipeline decision:
│   │                          filter cutoffs, window size, label thresholds,
│   │                          split protocol, random seed, device selection.
│   │                          Change values HERE, not inline in other scripts.
│   ├── data_loader.py        ← loads DREAMER.mat → clean per-subject Python objects.
│   │                          (Phase 1 will add: preprocessing.py, labeling.py,
│   │                          features.py, datasets.py for PyTorch Dataset classes)
│   │                          (Phase 4 will add: gan.py for CWGAN-GP)
│   │                          (Phase 2/5 will add: models.py, train.py, evaluate.py)
│   │
│   └── (future files land here as we build each phase — keeps all logic in
│         one importable package instead of scattered notebooks)
│
├── notebooks/               ← EXPLORATION ONLY. Jupyter notebooks for:
│                                - visually inspecting raw/filtered EEG signals
│                                - plotting label distributions, class balance
│                                - t-SNE/UMAP plots for GAN validation
│                                - ad-hoc debugging
│                              Nothing in here should be required for the
│                              pipeline to run — if a notebook produces
│                              something the pipeline depends on, that logic
│                              belongs in src/ instead.
│
├── models/                   ← SAVED MODEL WEIGHTS (checkpoints).
│                                e.g. models/cwgan_gp_generator.pt,
│                                models/eegnet_fold3.pt
│                                Not committed to git if large — see .gitignore note below.
│
└── outputs/                  ← RESULTS: metrics tables (csv), confusion
                                 matrices, plots, the actual numbers/figures
                                 that go into your paper. Anything you'd want
                                 to screenshot or copy into a LaTeX/Word doc
                                 for the paper lives here.
```

## Recommended .gitignore (if you haven't set one up)

```
data/raw/*
data/processed/*
models/*.pt
models/*.pth
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Raw data and trained weights are large and shouldn't go in git — keep the
repo to code only, and note in this README where to download DREAMER.mat from
so anyone (including future-you on the other machine) can regenerate
everything.

## Setup

```bash
pip install -r requirements.txt
```

## Quick sanity check

```bash
python -m src.data_loader
```

Should load DREAMER.mat and print channel counts, clip length ranges, and
subject 1's valence/arousal scores. If this fails, check that DREAMER.mat is
actually at `data/raw/DREAMER.mat`.

## Pipeline phases (see project plan for full detail)

- **Phase 0** — label definition, split protocol (locked in `config.py`)
- **Phase 1** — preprocessing (bandpass filter, baseline correction, epoching, normalization)
- **Phase 2** — baseline classification on real data only (all 6 models)
- **Phase 3** — class imbalance check
- **Phase 4** — CWGAN-GP training (feature-level augmentation)
- **Phase 5** — re-run classifiers with augmented training data
- **Phase 6** — analysis, comparison tables, writeup

## Datasets

- **DREAMER** (primary) — 14ch Emotiv EPOC, valence/arousal/dominance self-report,
  used as a proxy stress label via the arousal-valence quadrant model.
  https://zenodo.org/records/546113
- **STEW** (planned addition) — 14ch Emotiv EPOC, direct 0-9 workload/stress rating.
- **DASPS** (planned addition) — 14ch Emotiv EPOC+, HAM-A anxiety-labeled.

Note: all three use proxy or adjacent constructs for "stress" (emotion,
workload, anxiety respectively) — this is stated explicitly in the paper's
limitations, not something to gloss over.
