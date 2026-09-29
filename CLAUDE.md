# CLAUDE.md — project context for AI assistants and collaborators

Read this first in every new session. Then read `docs/RESULTS.md` (all real
numbers so far) and `paper/PAPER_DRAFT.md` (the manuscript being written).

## What this project is

Subject-independent EEG mental-workload classification on the **STEW** dataset
(48 subjects, Emotiv EPOC 14 channels, 128 Hz; rest vs. SIMKAP multitasking),
with a research paper as the goal. Three studies:

1. **Benchmark** — 25 models (8 classical/Riemannian, 17 deep) on identical
   subject-independent folds.
2. **Augmentation study** — 8 standard EEG augmentations vs. CWGAN-GP vs.
   conditional DDPM, across training-subject budgets (6, 12, 24, all).
3. **Validity** — task-vs-rest band-power topomaps and artifact-robustness
   ablations (no gamma; F7/F8/T7/T8 zeroed; both).

Owner: Aryan Kalra (VIT). STEW only — other datasets were deliberately removed.

## Code map

```
src/stewbench/
  constants.py        STEW facts, channel order/positions, bands
  settings.py         YAML-backed config dataclasses (every result-affecting number)
  data/raw.py         loads subXX_lo/hi.txt (searches data/raw/stew recursively)
  data/preprocess.py  band-pass -> windows -> class-agnostic MAD artifact rejection; cache
  data/inputs.py      per-window z-score; optional Euclidean Alignment (transductive)
  splits.py           subject folds (fixed by fold_seed) + subject-disjoint inner validation
  features.py         spectral features for classical models
  models/             registry (__init__.py), classical.py, convolutional.py,
                      recurrent.py, attention.py, graph.py, common.py
  training.py         one uniform training recipe for all deep models
  augment/            transforms.py (online), cwgan.py, ddpm.py, quality.py
  evaluation/         metrics.py (window/recording/subject, bootstrap CIs),
                      stats.py (Friedman/Nemenyi/Wilcoxon/Holm), complexity.py
  artifacts.py        Model E artifact bank + counterfactual generator
  experiments/        benchmark.py, augmentation.py, neuro.py, robust.py (Model E), common.py
  reporting/          tables (csv/tex/md), figures (pdf/png), report.py
  cli.py              python -m stewbench {check,preprocess,neuro,benchmark,augment,report,smoke,models}
configs/              benchmark*.yaml, augmentation.yaml, ablation_*.yaml, model_e_{dev,final}.yaml
docker/               Dockerfile (NGC PyTorch base) + run.sh helper
docs/                 NVIDIA_NGC_GUIDE.md, RESEARCH_DESIGN.md, RESULTS.md, PRIOR_ART.md, MODEL_E_PLAN.md
paper/                PAPER_DRAFT.md
tests/                pytest suite (fixture-based; slow tests train models on CPU)
```

## How experiments are run

All GPU work runs in Docker on the lab workstation (RTX A4000 16 GB, driver
580, image built from `nvcr.io/nvidia/pytorch:26.07-py3`, which runs in CUDA
forward-compatibility mode and was verified working). Lab rule: never install
software on the host; everything goes through Docker.

```bash
docker/run.sh build                      # once
docker/run.sh check                      # GPU + 48 subjects
DETACH=1 NAME=<name> docker/run.sh benchmark --config configs/<file>.yaml
docker logs --tail 20 <name>             # progress; runs are resumable
```

Outputs go to `outputs/stew/<experiment>/` (git-ignored; copy numbers into
`docs/RESULTS.md`). Data lives in `data/raw/stew/` (git-ignored).

## Non-negotiable protocol rules (do not break these)

- Split by **subject**, never by window. Test subjects are touched once.
- Early stopping / tuning only on the subject-disjoint inner-validation pool.
- No per-recording normalization (each STEW recording is one class → label leak).
- Transductive methods (re-centred Riemannian, Euclidean Alignment) are marked †
  and reported separately.
- Generators are trained inside each fold on inner-training subjects only.
- Report all seeds (mean ± SD), never the best seed. Never tune on test results.
- Numbers from `smoke` / the synthetic fixture are fake and must never be reported.

## Current status (update this section after every session)

- [x] Main benchmark (`benchmark_main`, 25 models × 5 seeds × 10 folds)
- [x] Neurophysiology topomaps
- [x] Augmentation study (468 runs)
- [x] Ablation: artifact_robust (no gamma + F7/F8/T7/T8 zeroed)
- [x] Ablation: no_gamma
- [x] Ablation: no_frontotemporal (drop is entirely from F7/F8/T7/T8; ranking flips to deep models)
- [ ] **Model E (artifact-counterfactual consistency training)**: implemented
      (`src/stewbench/artifacts.py`, `experiments/robust.py`, plan in `docs/MODEL_E_PLAN.md`).
      Dev rounds 1-2 done (RESULTS.md §5). Design FROZEN 2026-09-29: acct_cd@3 (consistency weight 3 +
      channel dropout). Next: run `configs/model_e_final.yaml` ONCE (test folds, 5 seeds, EEGNet + ATCNet).
      Do not change the design after seeing final numbers.
- [ ] Second dataset (e.g. COG-BCI) as untouched confirmation
- [ ] Optional: LOSO benchmark (`configs/benchmark_loso.yaml`) for literature comparison
- [ ] Paired per-subject test main vs. ablations (not implemented yet)
- [ ] Paper: finish Methods numbers (window counts, rejection rates), Discussion, figures
- [ ] Patent: not claimed; any novel method must be discussed with the VIT IPR office
      before any public disclosure (preprint, public repo)

## Conventions

- Development branch: `claude/wizardly-einstein-11gv8x`.
- Keep `docs/RESULTS.md` as the single source of truth for numbers; the paper
  cites values from there.
- Figures: family colours fixed in `reporting/style.py`; PDF + 300-dpi PNG.
