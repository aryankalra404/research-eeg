# Research design, literature and paper checklist

## Research questions

- **RQ1 — Benchmark.** Under a strict subject-independent protocol, how do
  classical, Riemannian and deep EEG models compare for rest vs. SIMKAP
  workload decoding on STEW, in accuracy, calibration and cost?
- **RQ2 — Augmentation.** Does data augmentation (standard transforms, a
  CWGAN-GP, a conditional diffusion model) improve cross-subject decoding?
- **RQ3 — Data efficiency.** Does any benefit grow as fewer training subjects
  are available (6, 12, 24, all)?
- **RQ4 — Validity.** Do the data and the models reflect known workload
  physiology (frontal theta increase, posterior alpha decrease)?

## Why the GAN is still worth studying (STEW is balanced)

STEW has equal rest and task data, so a GAN is **not** needed for class
balancing. The legitimate question is whether synthetic data helps a model
generalize to *new people* when few training subjects exist — the real
bottleneck in EEG (48 subjects). Reported GAN/diffusion gains in the
literature are mostly largest in low-data regimes and are rarely compared
against simple augmentations under subject-independent evaluation. This repo
therefore:

1. trains every generator inside each fold on training subjects only;
2. compares it against 8 standard augmentations (Rommel et al., 2022) and a
   diffusion model, at matched synthetic fractions;
3. sweeps the number of training subjects;
4. reports synthetic-data fidelity, diversity, utility (TSTR) and a
   memorization check.

A null or negative result here is still publishable and useful; do not tune
the augmentation on test results to force a gain.

## Evaluation pitfalls this protocol avoids

- **Window leakage.** Overlapping windows of the same person in train and
  test inflate accuracy dramatically; splits are always by subject.
- **Recording-wise normalization.** Each STEW recording is one class, so
  normalizing per recording leaks the label. Only per-window z-scoring is used.
- **Selecting on the test set.** Early stopping and hyper-parameters use a
  subject-disjoint inner-validation pool.
- **Treating windows as independent.** CIs resample whole subjects;
  model comparisons use per-subject scores (n = 48).
- **Transductive methods.** Re-centring / Euclidean Alignment use unlabelled
  test-subject data and are flagged (†) and reported separately.
- **Single seed.** 5 seeds; mean ± SD reported, never the best seed.

Many published STEW accuracies above 90% use window-level random splits;
compare only with papers using leave-subject-out protocols (use
`configs/benchmark_loso.yaml`) and say so explicitly.

## What to report (checklist)

- Dataset: 48 subjects, 14 channels, 128 Hz, labels = experimental condition
  (not self-rated workload), preprocessing, rejected-window rates per class.
- Protocol: folds, inner validation, seeds, training recipe, no test tuning.
- Table 1 (main results), Table 2 (secondary/calibration), Table 3 (cost),
  Table 4 (pairwise tests vs. best), CD diagram, per-subject distribution.
- Augmentation table, forest plot, data-efficiency curve, synthetic quality
  table, real-vs-synthetic PSD.
- Neurophysiology topomaps; channel/band importance.
- Limitations: consumer-grade 14-channel headset, single session, rest vs.
  task contrast (also differs in visual input and motor activity), condition
  labels rather than graded workload, window-level artifact rejection only.
- Code, configs and manifests (git revision, versions, data checksum).

## On patents

A benchmark, standard baselines and a standard GAN are not novel inventions.
Patentability would need a new technical method (e.g. a new generator or
training scheme with a demonstrated, reproducible gain under this protocol).
Consult your institution's technology-transfer office or a patent attorney
before publishing if you want to protect such a method, since public
disclosure (including a preprint or public repo) can affect patent rights.

## Key references

- Lim, Sourina & Wang. STEW: Simultaneous Task EEG Workload data set. *IEEE TNSRE* 26(11), 2018.
- Lawhern et al. EEGNet. *J. Neural Eng.* 15(5), 2018.
- Schirrmeister et al. Deep learning with CNNs for EEG decoding and visualization. *Hum. Brain Mapp.* 38(11), 2017.
- Ingolfsson et al. EEG-TCNet. *IEEE SMC*, 2020.
- Ding et al. TSception. *IEEE Trans. Affective Computing*, 2023.
- Song et al. EEG Conformer. *IEEE TNSRE* 31, 2023.
- Altaheri et al. ATCNet (physics-informed attention TCN). *IEEE Trans. Ind. Inform.* 19(2), 2023.
- Song et al. DGCNN for EEG emotion recognition. *IEEE Trans. Affective Computing*, 2018.
- Barachant et al. Multiclass BCI classification by Riemannian geometry. *IEEE TBME* 59(4), 2012.
- Zanini et al. Transfer learning: a Riemannian geometry framework. *IEEE TBME* 65(5), 2018.
- He & Wu. Transfer learning for BCIs: a Euclidean space data alignment approach. *IEEE TBME* 67(2), 2020.
- Lotte et al. A review of classification algorithms for EEG-based BCI: a 10-year update. *J. Neural Eng.* 15(3), 2018.
- Gulrajani et al. Improved training of Wasserstein GANs. *NeurIPS*, 2017.
- Hartmann, Schirrmeister & Ball. EEG-GAN. arXiv:1806.01875, 2018.
- Ho, Jain & Abbeel. Denoising diffusion probabilistic models. *NeurIPS*, 2020.
- Rommel et al. Data augmentation for learning predictive models on EEG: a systematic comparison. *J. Neural Eng.* 19(6), 2022.
- Naeem et al. Reliable fidelity and diversity metrics for generative models. *ICML*, 2020.
- Guo et al. On calibration of modern neural networks. *ICML*, 2017.
- Demšar. Statistical comparisons of classifiers over multiple data sets. *JMLR* 7, 2006.
- Pope, Bogart & Bartolome. Biocybernetic system evaluates indices of operator engagement. *Biol. Psychol.* 40, 1995.
- Klimesch. EEG alpha and theta oscillations reflect cognitive and memory performance. *Brain Res. Rev.* 29, 1999.

Verify every citation's details against the publisher before submission.
