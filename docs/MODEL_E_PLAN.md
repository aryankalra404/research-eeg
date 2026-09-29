# Model E: artifact-counterfactual consistency training (ACCT)

Status: in development (started 2026-09-28). Design choices were made by the
project owner (defaults accepted: all artifact types, EEGNet base, goal =
robustness at similar accuracy). Record every design decision and date in the
lab notebook: inventorship requires human conception.

## Problem (from our own results)
On STEW, eye-adjacent channels (F7/F8/T7/T8) show broadband task effects, and
removing them costs 2-6 BAcc points. Classifiers partly learn ocular/EMG
shortcuts. Consumer headsets (Emotiv, no EOG) cannot measure these artifacts
directly, and VR headsets worn on top add more.

## Method
1. **Artifact bank (per fold, training subjects only, class-agnostic).**
   - Ocular: principal spatial components of the <4 Hz content with frontal
     dominance (AF3/AF4/F7/F8/F3/F4), yielding blink-like (symmetric) and
     saccade-like (F7 vs F8 antisymmetric) topographies, plus a bank of their
     real time courses.
   - EMG: channel profile of excess 20-45 Hz power; generated as band-limited
     noise with a smooth random envelope.
2. **Counterfactual windows.** For each training window, create a copy with
   ocular artifacts added (sampled topography × real waveform), ocular
   subspace removed from the low-frequency content, and/or EMG added.
   Operations are sampled independently of the label, then each window/channel
   is re-z-scored.
3. **Consistency loss.** Cross-entropy on original and counterfactual plus
   λ × symmetric KL between their predictions (λ = 1).

## Conditions compared (same folds, seeds, recipe)
`erm` (plain), `channel_dropout` (best simple augmentation), `cf_aug`
(counterfactuals, no consistency term), `acct` (full method).

## Metrics
- Clean BAcc / AUC / κ (standard).
- **Artifact-channel reliance:** BAcc with F7/F8/T7/T8 zeroed at test time,
  and the drop from clean.
- **Stress test:** BAcc and prediction flip rate with injected artifacts
  (bank from training subjects) at magnitudes 1, 2 and 3.
- **Shortcut ceiling:** an "eye-adjacent-channel expert" (spectral LR on
  F7/F8/T7/T8 + gamma features only).
- Paired Wilcoxon per subject vs. `erm` (Holm).

## Protocol
- `dev_mode: true` (configs/model_e_dev.yaml): all design decisions use
  inner-validation subjects only; test subjects are not evaluated.
- Final: `configs/model_e_final.yaml`, run once, plus LOSO.
- Later: second dataset (e.g. COG-BCI) as an untouched confirmation;
  VR/Emotiv extension with Quest 3S head-motion (IMU) as artifact ground truth.

## Patent notes
Candidate technical effect: reliable workload estimation on a consumer EEG
headset without EOG, robust to ocular/EMG and headset-induced artifacts.
Do not publish or demo before a provisional filing via the VIT IPR office.

## Decision log
- 2026-09-28: defaults chosen (all artifact types, EEGNet, robustness goal).
- 2026-09-28: round 1 (dev) — consistency term is the active ingredient; channel dropout complementary.
- 2026-09-29: round 2 (dev) — acct_cd@3 best on clean BAcc, AUC, ECE and stress; natural-artifact
  gap unchanged. Design frozen as acct_cd@3; final run with EEGNet + ATCNet (pre-declared).
