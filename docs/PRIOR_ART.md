# Prior-art notes (2026-09-28)

Collected by three parallel web searches from snippets only; arXiv, PMC, Frontiers
and Google Patents were not fetchable. **Every item must be verified in full text,
and a professional patent search (incl. CNIPA / Espacenet / InPASS) done before
filing.**

## 1. Candidate A — spectral-prior net + subject- and artifact-adversarial heads: largely ANTICIPATED

| Element | Prior art |
|---|---|
| Learnable band filters initialised to canonical bands, **on STEW** | EEGminer (Ludwig et al., J. Neural Eng. 2024) https://iopscience.iop.org/article/10.1088/1741-2552/ad44d7 |
| Learnable band-pass front-ends | Sinc-ShallowNet (Neural Netw. 2020), Sinc-EEGNet (arXiv 2101.10846), SincMSNet (JNE 2023), GREEN (Patterns 2025) |
| Filter bank + variance/band-power layer | FBCNet (arXiv 2104.01233) |
| Band tokens → transformer | BandVQ (arXiv 2605.24921), SPD band tokens (arXiv 2601.21521) |
| Subject-adversarial EEG (gradient reversal); side effect: blink/jaw artifacts suppressed | Özdenizci et al., IEEE Access 2020 https://ieeexplore.ieee.org/document/8981912/ |
| Multiple adversarial nuisance heads | DANN-MAT (ESWA 2024), AutoBayes (arXiv 2007.01255) |
| Adversary on non-subject nuisance | Özdenizci 2019 (drowsiness) arXiv 1907.09540 |
| Continuous-confound adversary | BR-Net (Adeli et al.) https://openreview.net/forum?id=Bke8764twr |
| EOG estimated from forehead EEG during MATB workload | IEEE 9806494 |
| Confound-invariant deep EEG (patent) | WO2023240056A1 |
| Cross-subject workload DA/DG | CIDGN (2026), CDBG (arXiv 2609.30831), ProtoGIB-Workload (arXiv 2608.10647), bi-classifier DA (PubMed 39801913), AdaptEEG (PubMed 40030419), EA + periodic/aperiodic features (PubMed 40039636) |

## 2. Patents seen (snippets only)
- US20160113539A1, US11154229B2 (TCS): Emotiv-class headset, artifact filtering, alpha/theta power → cognitive load.
- US7865235B2, US20080218472A1 (Emotiv): mental-state classification from headset EEG.
- US20070135727A1 / US7684856: frontal artifact detection without EOG.
- US8364255B2, US9089310B2 (BrainScope): frontal EEG artifact removal.
- US10945654B2 (Muse): self-calibrating neurofeedback with ML state classification.
- WO2015040532A2, WO2016162820A1, US7580742B2: EEG cognitive load / task classification.
- US12,424,316: cross-session EEG domain adaptation (brainprint).
- US12,431,117: GAN-generated EEG; US20230121812A1: generic GAN augmentation.
- WO2023240056A1: deep EEG mapped to confound-inhibited subspace.
- No patent found claiming artifact-counterfactual consistency training or subject-selected synthetic "virtual subjects" for EEG workload (absence of evidence only).

## 3. Artifact confound evidence
- Saccadic spike potentials masquerade as gamma (Yuval-Greenberg et al., Neuron 2008).
- Eye-movement confounds in working-memory decoding (eNeuro 2018); horizontal eye movement alone gives 85.6% in handwriting EEG decoding (arXiv 2605.15698).
- Task-vs-rest alpha confounded by eye state/visual input (Applied Neuropsychology 2026).
- Only in-fold confound regression is valid (Snoek et al., bioRxiv 290684).

## 4. Generative augmentation
- Trust-gated augmentation (npj Digit. Med. 2026): teacher filters synthetic windows + fail-closed validation gate.
- TarDiff (utility-guided diffusion), EEGDiffuser, WGAN-GP vs diffusion fidelity/utility benchmark (arXiv 2509.08188), diffusion + DANN cross-subject MI (Brain Informatics 2026), FBGAN / TRANSIT-EEG (subject-specific generation).

## 5. Can LLMs generate novel ideas? Inventorship
- Si, Yang & Hashimoto (ICLR 2025): LLM ideas rated more novel than experts' on paper, slightly less feasible; only ~5% of seeds non-duplicate.
- Execution follow-up (arXiv 2506.20803): after 100+ h execution each, LLM ideas' scores dropped more; ranking flipped.
- Gupta & Pruthi (ACL 2025): 24% of 50 AI-generated research documents were paraphrases of existing work.
- DABUS: AI cannot be an inventor (US Fed. Cir. 2022; UK SC 2023; Indian Patent Office refusal Apr 2026).
- USPTO revised guidance (Nov 2025): AI is a tool; the human must conceive the complete claimed invention.
- India, Patents Act s.3(k): algorithms / computer programs *per se* are not patentable; claims must show a technical effect (e.g. device-level robustness).
