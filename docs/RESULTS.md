# Results log (real STEW data)

Single source of truth for numbers reported in the paper. Copied from
`outputs/stew/*/report/` on the lab workstation. BAcc = balanced accuracy on
held-out subjects (window level) unless noted. Chance = 0.50.

## 1. Main benchmark — `benchmark_main` (completed 2026-09-25)

10-fold subject-independent CV, 5 seeds, identical folds for all models.
Classical models are deterministic given the folds (SD = 0 across seeds is
expected; only RF varies).

| Model | Family | BAcc (mean ± SD) | BAcc 95% CI | ROC-AUC | κ | ECE | Rec. Acc |
|---|---|---|---|---|---|---|---|
| Spectral + LR | Feature | 0.829 | [0.796, 0.861] | 0.904 | 0.658 | 0.022 | 0.938 |
| Spectral + SVM-RBF | Feature | 0.829 | [0.791, 0.865] | 0.903 | 0.657 | 0.018 | 0.906 |
| Spectral + sLDA | Feature | 0.827 | [0.795, 0.858] | 0.905 | 0.655 | 0.061 | 0.958 |
| Riemann re-centred TS + LR † | Riemannian | 0.825 | [0.778, 0.865] | 0.895 | 0.650 | 0.029 | 0.979 |
| Spectral + RF | Feature | 0.821 ± 0.002 | [0.786, 0.854] | 0.897 | 0.641 | 0.097 | 0.927 |
| Spectral + GBDT | Feature | 0.805 | [0.769, 0.840] | 0.893 | 0.609 | 0.037 | 0.917 |
| ATCNet | EEG Transformer | 0.803 ± 0.010 | [0.764, 0.841] | 0.882 | 0.605 | 0.062 | 0.898 |
| EEGNet-8,2 | EEG CNN | 0.801 ± 0.014 | [0.763, 0.839] | 0.872 | 0.602 | 0.044 | 0.900 |
| EEG-TCNet | EEG CNN | 0.797 ± 0.004 | [0.757, 0.836] | 0.877 | 0.594 | 0.054 | 0.879 |
| Riemann TS + LR | Riemannian | 0.795 | [0.757, 0.831] | 0.875 | 0.590 | 0.040 | 0.885 |
| TSception | EEG CNN | 0.794 ± 0.006 | [0.756, 0.830] | 0.877 | 0.587 | 0.030 | 0.885 |
| ShallowConvNet | EEG CNN | 0.785 ± 0.012 | [0.750, 0.819] | 0.860 | 0.570 | 0.057 | 0.871 |
| 1D-CNN | Generic CNN | 0.784 ± 0.004 | [0.747, 0.819] | 0.855 | 0.567 | 0.049 | 0.875 |
| EEG Conformer | EEG Transformer | 0.778 ± 0.005 | [0.739, 0.815] | 0.858 | 0.553 | 0.135 | 0.873 |
| CNN-LSTM | Generic CNN | 0.776 ± 0.012 | [0.742, 0.811] | 0.831 | 0.552 | 0.071 | 0.873 |
| GRU | Recurrent | 0.766 ± 0.005 | [0.730, 0.801] | 0.830 | 0.531 | 0.057 | 0.854 |
| BiLSTM | Recurrent | 0.765 ± 0.005 | [0.731, 0.798] | 0.827 | 0.529 | 0.045 | 0.871 |
| LSTM | Recurrent | 0.763 ± 0.014 | [0.732, 0.792] | 0.824 | 0.525 | 0.035 | 0.902 |
| DGCNN | Graph | 0.761 ± 0.010 | [0.725, 0.796] | 0.832 | 0.522 | 0.047 | 0.885 |
| Swin (STFT) | Vision Transformer | 0.740 ± 0.013 | [0.706, 0.774] | 0.817 | 0.480 | 0.082 | 0.844 |
| ViT (STFT) | Vision Transformer | 0.734 ± 0.008 | [0.699, 0.768] | 0.805 | 0.468 | 0.066 | 0.827 |
| Electrode GCN | Graph | 0.724 ± 0.009 | [0.685, 0.762] | 0.799 | 0.446 | 0.034 | 0.819 |
| Riemann MDM | Riemannian | 0.713 | [0.665, 0.762] | 0.783 | 0.426 | 0.191 | 0.729 |
| Vanilla RNN | Recurrent | 0.703 ± 0.009 | [0.677, 0.730] | 0.753 | 0.407 | 0.034 | 0.867 |
| DeepConvNet | EEG CNN | 0.633 ± 0.087 | [0.591, 0.678] | 0.769 | 0.268 | 0.216 | 0.656 |

- Friedman χ² = 283.00, p = 4.3e-46; Nemenyi CD = 5.50 ranks (25 models, 48 subjects).
- Wilcoxon vs. Spectral + LR (Holm over all pairs): significantly worse only for
  Riemann MDM (p_Holm = .0093), Vanilla RNN (6.5e-7), LSTM (.016), DeepConvNet (9e-9),
  ViT (.001), Swin (.00051), Electrode GCN (.0004), DGCNN (.03). All others n.s.
- Deep models reach their best inner-validation epoch very early (median 1–3 for
  ShallowConvNet, 1D-CNN, ViT, TSception) → rapid overfitting to training subjects.
- DeepConvNet is unstable across seeds (some seeds collapse) under the shared recipe.
- Complexity (params / MFLOPs / GPU ms): EEGNet 1,842 / 7.7 / 0.23; EEG-TCNet 3,814 /
  2.2 / 0.46; ATCNet 112,650 / 11.8 / 2.01; EEG Conformer 745,506 / 44.4 / 0.89.

## 2. Neurophysiology — task vs. rest (paired t, n = 48, BH-FDR over 14 × 5)

| Channel | Band | t | d_z | q |
|---|---|---|---|---|
| F8 | theta | +8.61 | +1.24 | 2.2e-09 |
| FC6 | theta | +6.68 | +0.96 | 8.8e-07 |
| F8 | delta | +5.87 | +0.85 | 9.9e-06 |
| F8 | gamma | +5.78 | +0.83 | 1.0e-05 |
| F8 | beta | +5.70 | +0.82 | 1.1e-05 |
| T8 | theta | +5.59 | +0.81 | 1.3e-05 |
| F7 | gamma | +4.94 | +0.71 | 1.0e-04 |
| O2 | theta | +4.43 | +0.64 | 4.0e-04 |
| AF4 | theta | +3.93 | +0.57 | 1.5e-03 |
| P8 | alpha | −3.02 | −0.44 | 0.016 |
| F4 | alpha | −2.99 | −0.43 | 0.016 |
| O2 | alpha | −2.88 | −0.42 | 0.021 |

Interpretation: frontal theta increase and posterior alpha decrease (expected
workload signatures); broadband increases at F7/F8/T8 suggest ocular/EMG
contributions (eye-adjacent / temporal-muscle sites).

## 3. Artifact-robustness ablations (3 seeds, same folds)

| Model | Main | No gamma (0.5–30 Hz) | F7/F8/T7/T8 zeroed + no gamma | No F7/F8/T7/T8 only |
|---|---|---|---|---|
| Spectral + SVM | 0.829 | 0.825 | 0.785 | TODO |
| Spectral + LR | 0.829 | 0.824 | 0.776 | TODO |
| Spectral + sLDA | 0.827 | 0.819 | 0.769 | TODO |
| Riemann TS + LR | 0.795 | 0.799 | 0.777 | TODO |
| ATCNet | 0.803 | 0.797 | 0.777 | TODO |
| EEGNet-8,2 | 0.801 | 0.795 | 0.770 | TODO |
| EEG-TCNet | 0.797 | 0.789 | 0.769 | TODO |
| TSception | 0.794 | 0.805 | 0.770 | TODO |
| ShallowConvNet | 0.785 | 0.786 | 0.759 | TODO |

- Artifact-robust Friedman χ² = 13.59, p = 0.093 → models no longer differ.
- Gamma removal changes BAcc by ≤ 1 point; the drop comes from the eye-adjacent
  channels. Main-vs-ablation comparisons use slightly different window sets
  (rejection depends on the channels kept).

## 4. Augmentation study (3 seeds, 10 folds, generators trained per fold)

### All training subjects (Δ = paired per-subject vs. none; Holm within classifier)

| Classifier | none | best online aug (Δ) | CWGAN-GP 0.5 / 1.0 (Δ) | DDPM 0.5 / 1.0 (Δ) |
|---|---|---|---|---|
| EEG Conformer | 0.776 | channel dropout 0.801 (+0.027*), mixup 0.798 (+0.023*), noise 0.792 (+0.017*) | +0.012 / +0.008 (n.s.) | −0.004 / +0.005 (n.s.) |
| EEGNet-8,2 | 0.801 | sign flip 0.811 (+0.011, n.s.) | **−0.021* / −0.033*** | +0.004 / −0.010 (n.s.) |
| ShallowConvNet | 0.794 | channel dropout 0.810 (+0.017, p_Holm = .10) | −0.015 (p = .06) / **−0.025*** | +0.000 / +0.007 (n.s.) |

\* Holm p < 0.05.

### Data efficiency (BAcc by training subjects per fold: 6 / 12 / 24 / all)

| Classifier | none | CWGAN-GP 1.0 | DDPM 1.0 | mixup | channel dropout | freq. shift |
|---|---|---|---|---|---|---|
| EEG Conformer | .632 / .668 / .773 / .776 | .661 / **.752** / .764 / .782 | .641 / **.733** / .773 / .780 | **.679** / .698 / .780 / .798 | .657 / .674 / .778 / .801 | **.698** / .686 / .762 / .789 |
| EEGNet-8,2 | .712 / .742 / .775 / .801 | .710 / .727 / .738 / .768 | .703 / .727 / .763 / .790 | .707 / .727 / .770 / .795 | .717 / .745 / .780 / .807 | .732 / .745 / .773 / .800 |
| ShallowConvNet | .733 / .743 / .774 / .794 | .735 / .756 / .777 / .769 | .726 / **.772** / .780 / .799 | .710 / .746 / .775 / .797 | .745 / .752 / .788 / .810 | .714 / .738 / .772 / .788 |

Significant (Holm) at small budgets: 12 subjects — Conformer + CWGAN 1.0 +0.082,
CWGAN 0.5 +0.051, DDPM 1.0 +0.064, DDPM 0.5 +0.055, mixup +0.029; ShallowConvNet
+ DDPM 1.0 +0.030; EEGNet + mixup −0.015. 6 subjects — Conformer + frequency
shift +0.066, mixup +0.047, FT surrogate −0.028; EEGNet + frequency shift
+0.021; ShallowConvNet + FT surrogate −0.023. 24 subjects — EEGNet + CWGAN 1.0 −0.035.

### Synthetic-data quality (mean over folds, seeds, classes)

| Budget | Generator | LSD (dB) | Corr. err. | MMD | Density | Coverage | Memorization ratio | TRTR | TSTR |
|---|---|---|---|---|---|---|---|---|---|
| 6 | CWGAN-GP | 2.79 | 0.139 | 0.074 | 0.648 | 0.325 | 0.981 | 0.690 | 0.603 |
| 6 | DDPM | 3.92 | 0.128 | 0.321 | 0.446 | 0.138 | 0.989 | 0.690 | 0.647 |
| 12 | CWGAN-GP | 2.71 | 0.085 | 0.046 | 0.732 | 0.365 | 0.952 | 0.713 | 0.582 |
| 12 | DDPM | 3.65 | 0.332 | 0.141 | 1.081 | 0.367 | 0.931 | 0.713 | 0.688 |
| 24 | CWGAN-GP | 2.85 | 0.077 | 0.037 | 1.068 | 0.488 | 0.935 | 0.747 | 0.599 |
| 24 | DDPM | 4.67 | 0.362 | 0.211 | 0.985 | 0.403 | 0.915 | 0.747 | 0.709 |
| all | CWGAN-GP | 2.55 | 0.066 | 0.034 | 1.278 | 0.572 | 0.941 | 0.760 | 0.600 |
| all | DDPM | 5.12 | 0.364 | 0.246 | 0.997 | 0.453 | 0.907 | 0.760 | 0.707 |

Fidelity ≠ utility: CWGAN-GP matches spectra/covariance better, DDPM carries more
class information (TSTR 0.71 vs 0.60).

## TODO numbers for the paper
- Total windows after rejection and per-class rejection rates (`docker/run.sh preprocess`).
- `ablation_no_frontotemporal` results.
- Paired per-subject tests main vs. ablations.
