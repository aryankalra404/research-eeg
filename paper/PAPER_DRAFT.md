# Paper draft

> Working manuscript. Numbers come from `docs/RESULTS.md`; update both together.
> `[TODO]` marks missing items. Target venues to consider: IEEE TNSRE, Journal of
> Neural Engineering, Frontiers in Neuroergonomics, IEEE EMBC / SMC (shorter version).

## Title (candidates)

1. *Do Deep Networks and Generative Augmentation Help Cross-Subject EEG Workload
   Decoding? A Leakage-Free Benchmark on STEW*
2. *When Does Synthetic EEG Help? Generative Augmentation, Data Efficiency and
   Artifact Confounds in Cross-Subject Mental-Workload Classification*

## Abstract (draft, ~250 words)

Electroencephalography (EEG) is a promising signal for monitoring mental workload,
but reported accuracies on public datasets are often inflated by evaluation
protocols that let the same person appear in training and test data. We present
a leakage-free, subject-independent benchmark of 25 classifiers — spectral-feature
and Riemannian pipelines, recurrent networks, EEG-specific convolutional networks,
transformers and graph networks — on the Simultaneous Task EEG Workload (STEW)
dataset (48 subjects, rest vs. SIMKAP multitasking). All models share identical
subject-level folds, a uniform training recipe, subject-disjoint early stopping,
five seeds and subject-clustered statistics. Hand-crafted spectral features with a
linear classifier achieved the highest balanced accuracy (0.829, 95% CI
0.796–0.861), and the best deep networks (ATCNet 0.803, EEGNet 0.801) were not
significantly better. We then asked whether data augmentation, including a
conditional Wasserstein GAN and a conditional diffusion model trained strictly
within each fold, improves generalization to unseen people. With all training
subjects, generative augmentation did not help and the GAN significantly degraded
compact CNNs (−2 to −3 points), whereas simple channel dropout and mixup improved
a transformer (+2 to +3 points). With only 12 training subjects, synthetic data
substantially improved the transformer (+8.2 points GAN, +6.4 points diffusion).
Synthetic-data fidelity did not predict utility: the GAN matched EEG spectra more
closely, but diffusion samples carried more class information. Finally,
physiological analysis and ablations show that part of the rest-vs-task contrast
arises at eye-adjacent electrodes; removing them lowered accuracy by 2–6 points
but left all models well above chance. We release code, configurations and
protocols to support reproducible workload-decoding research.

## 1. Introduction

- Motivation: mental-workload monitoring (aviation, driving, human–computer
  interaction); EEG as a direct, portable measure; consumer headsets (Emotiv).
- Problem 1 — evaluation: many STEW results use window-level random splits, so
  overlapping windows from the same person appear in train and test, inflating
  accuracy (cite leakage studies). Cross-subject performance is what matters for
  deployment.
- Problem 2 — data scarcity: 48 subjects; deep models overfit; generative
  augmentation (GANs, diffusion) is widely proposed, but rarely evaluated under
  strict subject-independent protocols or against simple augmentation.
- Problem 3 — validity: rest vs. multitasking also differs in eye movements and
  muscle activity; few studies check whether classifiers exploit artifacts.
- Contributions:
  1. A leakage-free benchmark of 25 models with identical folds, uniform
     training, 5 seeds, subject-clustered CIs and nonparametric statistics.
  2. The finding that spectral features match or beat deep networks
     cross-subject on STEW.
  3. A controlled study of 8 conventional augmentations, a CWGAN-GP and a
     conditional DDPM across four training-set sizes, showing *when* synthetic
     EEG helps (data-hungry models, few subjects) and when it harms.
  4. Evidence that synthetic-data fidelity metrics do not predict utility.
  5. A physiological and artifact analysis quantifying the contribution of
     eye-adjacent channels.
  6. Open, reproducible code (Docker, configs, manifests).

## 2. Related work

- STEW dataset and prior results (Lim et al., 2018; later CNN/LSTM/hybrid work);
  note which protocols are subject-independent. `[TODO: table of prior STEW
  results with protocol column — only include papers whose protocol is verified]`
- EEG decoding architectures: EEGNet, Shallow/DeepConvNet, EEG-TCNet, TSception,
  EEG Conformer, ATCNet, DGCNN; Riemannian methods; spectral features.
- EEG data augmentation: Rommel et al. (2022) systematic comparison; GANs
  (Hartmann et al., 2018; WGAN-GP); diffusion models for EEG.
- Evaluation pitfalls: data leakage in EEG deep learning; statistical comparison
  of classifiers (Demšar, 2006).

## 3. Methods

### 3.1 Dataset
STEW: 48 subjects, Emotiv EPOC (14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8,
T8, FC6, F4, F8, AF4), 128 Hz, 2.5 min rest and 2.5 min SIMKAP multitasking per
subject. Labels are the experimental condition (rest = 0, task = 1), not
self-ratings (ratings unavailable for subjects 5, 24, 42).

### 3.2 Preprocessing
Zero-phase 4th-order Butterworth band-pass 0.5–45 Hz; 4-s windows with 50%
overlap; class-agnostic artifact rejection per subject (reject windows whose
peak-to-peak amplitude on any channel exceeds median + 60 × MAD computed over
both conditions); per-window, per-channel z-scoring for deep models (no
statistics cross window boundaries; per-recording normalization deliberately
avoided because each recording is one class). Windows after rejection: `[TODO]`;
rejection rate rest/task: `[TODO]`.

### 3.3 Evaluation protocol
10-fold subject-independent cross-validation with folds fixed by a single seed
and shared by all models. Inside each fold, 15% of training subjects form a
subject-disjoint inner-validation pool for early stopping and hyper-parameter
selection; test subjects are used once. Five training seeds (three for ablations
and augmentation). Metrics: balanced accuracy (primary), accuracy, macro-F1,
Cohen's κ, MCC, ROC-AUC, PR-AUC, sensitivity, specificity, Brier score, log-loss,
expected calibration error; recording-level accuracy (mean window probability
per recording). 95% CIs by subject-cluster bootstrap (2,000 resamples).
Statistics on per-subject balanced accuracy (n = 48): Friedman test with Nemenyi
critical difference; pairwise Wilcoxon signed-rank tests with Holm correction and
rank-biserial effect sizes.

### 3.4 Models
- Classical: spectral features (log absolute and relative band power in
  δ/θ/α/β/γ, θ/α ratio, engagement index β/(α+θ), hemispheric asymmetry, Hjorth
  parameters) with shrinkage LDA, logistic regression, RBF-SVM, random forest,
  gradient boosting; Riemannian MDM and tangent-space LR on OAS covariances;
  re-centred tangent space (transductive, †).
- Deep: 1D-CNN, vanilla RNN, LSTM, BiLSTM, GRU, CNN-LSTM, EEGNet-8,2,
  ShallowConvNet, DeepConvNet, EEG-TCNet, TSception, EEG Conformer, ATCNet,
  ViT and Swin on log-STFT, electrode GCN, DGCNN. Kernel lengths designed for
  250 Hz rescaled to 128 Hz.
- Uniform training: AdamW (lr 1e-3, weight decay 1e-3), batch 64, 3-epoch warm-up
  + cosine schedule, up to 100 epochs, early stopping on inner-validation loss
  (patience 15), class-weighted cross-entropy, gradient clipping 1.0, mixed
  precision. No per-model tuning.

### 3.5 Augmentation study
Classifiers fixed before the benchmark results were known: EEGNet,
ShallowConvNet, EEG Conformer. Online augmentations (p = 0.5): Gaussian noise,
smooth time mask, channel dropout, FT surrogate, frequency shift, sign flip, time
reversal, mixup. Generative: CWGAN-GP (Adam 1e-4, β = (0, 0.9), 5 critic steps,
λ_GP = 10, EMA generator) and conditional DDPM (1D U-Net, cosine schedule,
classifier-free guidance training, 50-step DDIM sampling), 300 epochs each,
trained per fold on inner-training subjects only; synthetic windows added at 50%
or 100% of each class's real count. Training-subject budgets 6, 12, 24 and all
(nested subsets; validation and test pools unchanged). Synthetic quality:
log-spectral distance, channel-correlation error, feature MMD, density/coverage,
TRTR/TSTR, nearest-neighbour memorization ratio.

### 3.6 Physiology and artifact analysis
Paired t-tests of per-subject log band power (task − rest), BH-FDR over
14 channels × 5 bands; ablations removing gamma (0.5–30 Hz), zeroing F7/F8/T7/T8,
or both. Model-agnostic channel-permutation and band-removal importance.

### 3.7 Implementation
PyTorch in the NVIDIA NGC container 26.07 on an RTX A4000 (16 GB); code,
configurations and run manifests (git revision, versions, data checksum)
released at `[TODO: public repository URL after IP review]`.

## 4. Results

### 4.1 Benchmark (Table 1, Fig. CD diagram, Fig. per-subject distribution)
- Spectral + LR 0.829 (CI 0.796–0.861), Spectral + SVM 0.829, sLDA 0.827; best
  deep: ATCNet 0.803 ± 0.010, EEGNet 0.801 ± 0.014, EEG-TCNet 0.797 ± 0.004.
- Friedman χ² = 283.0, p = 4.3e-46. After Holm correction the best model was not
  significantly better than any EEG-specific CNN, ATCNet, EEG Conformer, the
  recurrent models except LSTM and vanilla RNN, or 1D-CNN; it was significantly
  better than MDM, vanilla RNN, LSTM, DeepConvNet, ViT, Swin, electrode GCN and DGCNN.
- Recording-level accuracy 0.90–0.98 for the best models.
- Deep models selected very early epochs (median 1–3 for several), indicating
  rapid overfitting to training subjects; DeepConvNet was unstable across seeds.
- Calibration: feature models ECE ≈ 0.02; EEG Conformer 0.135.
- Cost: EEGNet has 1,842 parameters and runs in 0.23 ms per window on GPU,
  within 3 points of the best model. `[TODO: Pareto figure]`

### 4.2 Physiology (Fig. topomaps)
Task increased frontal theta (F8 t = 8.61, d_z = 1.24; FC6, AF4, F7, T8) and
decreased posterior alpha (P8, O2) — the canonical workload signature. F8/F7 and
T8 also showed broadband increases (delta to gamma), consistent with ocular and
muscle activity.

### 4.3 Artifact robustness (Table)
Removing gamma changed balanced accuracy by ≤ 1 point. Zeroing F7/F8/T7/T8 and
removing gamma reduced it by 2–6 points (spectral models −4 to −6; deep models
−2.4 to −3.1), yet all models stayed at 0.76–0.79 and no longer differed
significantly (Friedman p = 0.093). `[TODO: F7/F8/T7/T8-only ablation; paired
per-subject tests]`

### 4.4 Augmentation (Table 5, forest plot, data-efficiency figure)
- All subjects: CWGAN-GP reduced EEGNet (−0.021, −0.033) and ShallowConvNet
  (−0.025 at 100%); DDPM had no significant effect; channel dropout (+0.027),
  mixup (+0.023) and Gaussian noise (+0.017) improved EEG Conformer.
- 12 subjects: synthetic data improved EEG Conformer (CWGAN +0.082, DDPM +0.064)
  and DDPM improved ShallowConvNet (+0.030).
- 6 subjects: frequency shift (+0.066) and mixup (+0.047) improved EEG Conformer.
- No augmented deep model exceeded the unaugmented spectral baseline (best
  augmented deep ≈ 0.81 vs. 0.829).

### 4.5 Synthetic-data quality (Table 6)
CWGAN-GP had lower spectral distance (2.6 vs. 5.1 dB at full data) and MMD
(0.03 vs. 0.25), but lower TSTR balanced accuracy (0.60 vs. 0.71): higher
fidelity did not translate into class-discriminative utility. Memorization ratios
0.91–0.99 indicate no copying of training windows (heuristic).

## 5. Discussion

- Why features win cross-subject: relative band power and ratios normalize away
  inter-subject scale differences; deep models overfit subject identity with ~37
  training subjects.
- When synthetic EEG helps: data-hungry transformers in low-data regimes; not
  compact, strongly regularized CNNs; not when enough real subjects exist. The
  GAN's harm at full data coincides with its low TSTR — realistic-looking but
  weakly class-informative samples dilute the real signal.
- Practical recommendation: start with spectral features or EEGNet; use channel
  dropout; consider generative augmentation only for large models with few
  subjects, and validate utility (TSTR), not just fidelity.
- Validity: the workload signature is present, but eye-adjacent channels
  contribute several points of accuracy; future datasets should record EOG.

## 6. Limitations

Single dataset and session; consumer 14-channel headset without midline or EOG
channels; rest vs. multitasking also differs in visual and motor demands;
condition labels rather than graded workload; window-level artifact rejection
only (no ICA); uniform hyper-parameters may disadvantage some architectures
(e.g. DeepConvNet); three seeds in the augmentation study; Holm correction
applied within classifier and budget.

## 7. Conclusion
`[TODO after final results]`

## Figures and tables plan

| # | Content | Source file |
|---|---|---|
| Fig 1 | Pipeline and protocol diagram | `[TODO: draw]` |
| Fig 2 | Task − rest topomaps | `neurophysiology/figures/fig_task_vs_rest_topomaps.pdf` |
| Fig 3 | Critical-difference diagram | `benchmark_main/report/figures/fig_cd_diagram.pdf` |
| Fig 4 | Per-subject balanced accuracy | `.../fig_per_subject_bacc.pdf` |
| Fig 5 | Accuracy vs. parameters | `.../fig_bacc_vs_parameters.pdf` |
| Fig 6 | Data-efficiency curves | `augmentation_study/report/figures/fig_data_efficiency.pdf` |
| Fig 7 | Augmentation forest plot (all subjects) | `.../fig_augmentation_forest_budget_all.pdf` |
| Fig 8 | Real vs. synthetic spectra | `.../fig_psd_real_vs_cwgan_gp.pdf`, `..._ddpm.pdf` |
| Tab 1 | Main results | `benchmark_main/report/tables/table1_main_results.tex` |
| Tab 2 | Complexity | `.../table3_complexity.tex` |
| Tab 3 | Artifact ablations | from `docs/RESULTS.md` §3 |
| Tab 4 | Augmentation | `augmentation_study/report/tables/table5_augmentation.tex` |
| Tab 5 | Synthetic quality | `.../table6_synthetic_quality.tex` |
| Supp. | Secondary metrics, pairwise tests, channel/band importance, calibration | report folders |

## References
See `docs/RESEARCH_DESIGN.md` (verify every citation against the publisher before
submission; add the leakage-study and diffusion-EEG references).
