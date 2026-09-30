const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  HeadingLevel, AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat,
  Header, Footer, PageNumber, PageBreak, TableOfContents, VerticalAlign,
} = require("docx");

const DIR = __dirname;
const FONT = "Calibri";
const CONTENT = 9026; // A4 width minus 2 x 1 inch margins (DXA)
const ACCENT = "1F3A5F";
const HEAD_FILL = "E8EEF5";
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "B8C4D2" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };

// ---------- helpers ----------
const t = (text, opts = {}) => new TextRun({ text, font: FONT, ...opts });
const p = (children, opts = {}) =>
  new Paragraph({ children: typeof children === "string" ? [t(children)] : children,
                  spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(text)] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bullets", level: 0 },
  children: typeof runs === "string" ? [t(runs)] : runs, spacing: { after: 60, line: 264 } });
const numbered = (runs) => new Paragraph({ numbering: { reference: "numbers", level: 0 },
  children: typeof runs === "string" ? [t(runs)] : runs, spacing: { after: 60, line: 264 } });
const caption = (label, text) => new Paragraph({
  children: [t(label + " ", { bold: true, size: 19, color: ACCENT }), t(text, { size: 19, italics: true })],
  spacing: { before: 60, after: 200 }, alignment: AlignmentType.LEFT });
const tableCaption = (label, text) => new Paragraph({
  children: [t(label + " ", { bold: true, size: 19, color: ACCENT }), t(text, { size: 19 })],
  spacing: { before: 200, after: 80 }, keepNext: true });

function figure(file) {
  const data = fs.readFileSync(path.join(DIR, file));
  const widthPx = data.readUInt32BE(16), heightPx = data.readUInt32BE(20); // PNG IHDR
  const maxW = 600; // px at 96 dpi ~ 6.25 in
  const w = Math.min(maxW, widthPx);
  const h = Math.round(heightPx * (w / widthPx));
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file } })] });
}

function table(headers, rows, widths, { boldFirstCol = false, highlightRows = [], fontSize = 17 } = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, width, { header = false, bold = false, fill = null, align = AlignmentType.CENTER } = {}) =>
    new TableCell({
      borders: BORDERS, width: { size: width, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      shading: fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ alignment: align, spacing: { after: 0 },
        children: [t(String(text), { size: fontSize, bold: header || bold, color: header ? ACCENT : undefined })] })],
    });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((hd, i) =>
        cell(hd, widths[i], { header: true, fill: HEAD_FILL, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((v, i) =>
        cell(v, widths[i], { bold: (boldFirstCol && i === 0) || highlightRows.includes(ri),
          fill: highlightRows.includes(ri) ? "F4F8FC" : null,
          align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })),
    ],
  });
}
const note = (text) => new Paragraph({ children: [t(text, { size: 17, italics: true, color: "555555" })],
  spacing: { before: 60, after: 200 } });

// ---------- content ----------
const titlePage = [
  new Paragraph({ spacing: { before: 2200 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [t("PROGRESS REPORT", { size: 24, bold: true, color: "6B7C93", characterSpacing: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [t("Subject-Independent EEG Classification of Mental Workload on the STEW Dataset", { size: 40, bold: true, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [t("Baseline benchmark, artifact analysis, augmentation study and a proposed artifact-robust training method", { size: 24, italics: true, color: "444444" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [t("Aryan Kalra (25BCE0390)", { size: 24, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [t("Supervisor: Dr. Dhivyaa C R", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [t("Vellore Institute of Technology, Vellore", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1400 }, children: [t("29 September 2026", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, border: { top: { style: BorderStyle.SINGLE, size: 6, color: "B8C4D2", space: 8 } },
    children: [t("Confidential. This report describes an unpublished method that may be submitted for patent protection. Please do not circulate outside the supervisory team.", { size: 18, italics: true, color: "6B7C93" })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const toc = [
  new Paragraph({ children: [t("Contents", { size: 30, bold: true, color: ACCENT })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [t("Right-click and choose Update Field if the page numbers are missing.", { size: 16, italics: true, color: "888888" })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const benchRows = [
  ["Spectral + LR","Feature-based","0.829","[0.796, 0.861]","0.904","0.658","0.022","0.938"],
  ["Spectral + SVM (RBF)","Feature-based","0.829","[0.791, 0.865]","0.903","0.657","0.018","0.906"],
  ["Spectral + sLDA","Feature-based","0.827","[0.795, 0.858]","0.905","0.655","0.061","0.958"],
  ["Riemann re-centred TS + LR †","Riemannian","0.825","[0.778, 0.865]","0.895","0.650","0.029","0.979"],
  ["Spectral + RF","Feature-based","0.821 ± 0.002","[0.786, 0.854]","0.897","0.641","0.097","0.927"],
  ["Spectral + GBDT","Feature-based","0.805","[0.769, 0.840]","0.893","0.609","0.037","0.917"],
  ["ATCNet","EEG transformer","0.803 ± 0.010","[0.764, 0.841]","0.882","0.605","0.062","0.898"],
  ["EEGNet-8,2","EEG CNN","0.801 ± 0.014","[0.763, 0.839]","0.872","0.602","0.044","0.900"],
  ["EEG-TCNet","EEG CNN","0.797 ± 0.004","[0.757, 0.836]","0.877","0.594","0.054","0.879"],
  ["Riemann TS + LR","Riemannian","0.795","[0.757, 0.831]","0.875","0.590","0.040","0.885"],
  ["TSception","EEG CNN","0.794 ± 0.006","[0.756, 0.830]","0.877","0.587","0.030","0.885"],
  ["ShallowConvNet","EEG CNN","0.785 ± 0.012","[0.750, 0.819]","0.860","0.570","0.057","0.871"],
  ["1D-CNN","Generic CNN","0.784 ± 0.004","[0.747, 0.819]","0.855","0.567","0.049","0.875"],
  ["EEG Conformer","EEG transformer","0.778 ± 0.005","[0.739, 0.815]","0.858","0.553","0.135","0.873"],
  ["CNN-LSTM","Generic CNN","0.776 ± 0.012","[0.742, 0.811]","0.831","0.552","0.071","0.873"],
  ["GRU","Recurrent","0.766 ± 0.005","[0.730, 0.801]","0.830","0.531","0.057","0.854"],
  ["BiLSTM","Recurrent","0.765 ± 0.005","[0.731, 0.798]","0.827","0.529","0.045","0.871"],
  ["LSTM","Recurrent","0.763 ± 0.014","[0.732, 0.792]","0.824","0.525","0.035","0.902"],
  ["DGCNN","Graph","0.761 ± 0.010","[0.725, 0.796]","0.832","0.522","0.047","0.885"],
  ["Swin (STFT)","Vision transformer","0.740 ± 0.013","[0.706, 0.774]","0.817","0.480","0.082","0.844"],
  ["ViT (STFT)","Vision transformer","0.734 ± 0.008","[0.699, 0.768]","0.805","0.468","0.066","0.827"],
  ["Electrode GCN","Graph","0.724 ± 0.009","[0.685, 0.762]","0.799","0.446","0.034","0.819"],
  ["Riemann MDM","Riemannian","0.713","[0.665, 0.762]","0.783","0.426","0.191","0.729"],
  ["Vanilla RNN","Recurrent","0.703 ± 0.009","[0.677, 0.730]","0.753","0.407","0.034","0.867"],
  ["DeepConvNet","EEG CNN","0.633 ± 0.087","[0.591, 0.678]","0.769","0.268","0.216","0.656"],
];

const body = [
  // 1
  h1("1. Summary"),
  p("This report covers the work completed so far on cross-subject mental-workload classification from EEG, using the public STEW dataset (48 participants, 14-channel Emotiv EPOC, rest versus SIMKAP multitasking). The main aim of this phase was to build a reliable reference point before proposing anything new. For that reason most of the effort went into an evaluation pipeline in which no participant ever appears in both the training and the test data, and into testing a wide range of existing methods under exactly the same conditions."),
  p("Four results stand out at this stage:"),
  bullet([t("Baselines. ", { bold: true }), t("Across 25 models, the best balanced accuracy on unseen participants is 0.829 (spectral band-power features with logistic regression). The strongest deep networks, ATCNet (0.803) and EEGNet (0.801), are not significantly worse, but they are not better either.")]),
  bullet([t("Artifact confound. ", { bold: true }), t("Part of the rest-versus-task difference sits at the electrodes closest to the eyes (F7, F8, T7, T8). Removing these four channels lowers the feature-based models by 5 to 6 points and the deep models by only 1.4 to 2.1 points, after which the deep models rank first.")]),
  bullet([t("Synthetic data. ", { bold: true }), t("GAN and diffusion augmentation help a transformer when only 12 training participants are available (up to +8.2 points), but with the full training set the GAN lowers the accuracy of the compact CNNs.")]),
  bullet([t("Proposed method. ", { bold: true }), t("An artifact-counterfactual consistency training scheme (working name ACCT) improves EEGNet by 3.8 points on validation participants, halves its calibration error and reduces prediction flips under injected artifacts from 24.5% to 7.0%. These are development results; the final test on held-out participants has not been run yet.")]),

  // 2
  h1("2. Background and objectives"),
  p("Mental workload estimation from EEG has obvious uses in aviation, driving, surgery and other safety-critical settings. Consumer headsets such as the Emotiv EPOC make recording cheap, but they come with two practical problems: only a few channels are available, and there is no dedicated electrooculography (EOG) channel, so eye movements and muscle activity mix freely with brain activity."),
  p("Published results on STEW vary widely, and many of the high numbers (above 90%) come from splitting overlapping windows of the same recording between training and test sets. When a model sees a participant during training it can recognise that person rather than their workload, which inflates accuracy. The objectives for this project are therefore:"),
  numbered("Establish an honest, subject-independent benchmark of classical, Riemannian and deep-learning methods on STEW."),
  numbered("Check whether the data show the expected neurophysiological signs of workload, and how much of the signal is driven by ocular or muscle artifacts."),
  numbered("Test whether generative data augmentation (GAN and diffusion) improves generalisation to new participants."),
  numbered("Develop a training method that keeps accuracy while making the classifier less dependent on artifacts, with a view to publication and possible patent protection."),

  // 3
  h1("3. Dataset and preprocessing"),
  p("STEW contains 150 seconds of EEG at rest and 150 seconds during the SIMKAP multitasking test for each of 48 participants, sampled at 128 Hz from the 14 Emotiv channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4). The class label is the experimental condition. Self-reported workload ratings exist, but they are missing for three participants and were not used as labels. The preprocessing settings are listed in Table 1."),
  tableCaption("Table 1.", "Preprocessing settings (identical for every model)."),
  table(["Step", "Setting", "Reason"], [
    ["Band-pass filter", "0.5–45 Hz, 4th-order Butterworth, zero-phase", "Removes drift and line noise"],
    ["Windowing", "4 s windows, 50% overlap (512 samples)", "Enough length for theta/alpha estimates"],
    ["Artifact rejection", "Peak-to-peak > median + 60 × MAD, per participant", "Label-independent, same rule for both classes"],
    ["Normalisation", "z-score per window and channel", "No statistics shared across windows or recordings"],
    ["Excluded on purpose", "Per-recording normalisation", "Each recording is one class, so it would leak the label"],
  ], [2200, 3700, 3126]),

  // 4
  h1("4. Evaluation protocol"),
  p("All models were evaluated with the same 10-fold cross-validation over participants, so every participant is tested exactly once and never contributes training data to its own test fold. Inside each fold, 15% of the training participants form a separate validation pool that is used only for early stopping and hyper-parameter choices. Deep models share one training recipe (AdamW, learning rate 1e-3, batch size 64, cosine schedule with warm-up, up to 100 epochs, early stopping on validation loss) and none of them was tuned on test data. Every experiment was repeated with five random seeds."),
  p("Balanced accuracy is the main metric. We also report macro-F1, Cohen's kappa, MCC, ROC-AUC, PR-AUC, sensitivity, specificity, Brier score and expected calibration error (ECE), both per window and per recording (averaging the predictions over the 150-second recording). Confidence intervals come from a bootstrap that resamples whole participants, since windows from the same person are strongly correlated. Models are compared with the Friedman test, Nemenyi critical differences and Wilcoxon signed-rank tests on per-participant scores with Holm correction."),

  // 5
  h1("5. Baseline benchmark"),
  p("Table 2 lists all 25 baselines. The classical group uses spectral features (log and relative band power in the delta to gamma bands, theta/alpha ratio, engagement index, hemispheric asymmetry and Hjorth parameters) with five standard classifiers, plus three Riemannian-geometry pipelines. The deep group covers generic networks (1D-CNN, RNN, LSTM, BiLSTM, GRU, CNN-LSTM), EEG-specific networks (EEGNet, ShallowConvNet, DeepConvNet, EEG-TCNet, TSception, EEG Conformer, ATCNet), spectrogram transformers (ViT, Swin) and graph networks (electrode GCN, DGCNN)."),
  tableCaption("Table 2.", "Subject-independent benchmark on STEW (10-fold CV over participants, mean ± SD over 5 seeds). BAcc = balanced accuracy; Rec. Acc = recording-level accuracy; † uses unlabelled data of the test participant."),
  table(["Model", "Family", "BAcc", "95% CI", "AUC", "κ", "ECE", "Rec. Acc"], benchRows,
        [2350, 1450, 1050, 1250, 700, 700, 700, 826], { fontSize: 15, highlightRows: [0] }),
  note("Classical models are deterministic once the folds are fixed, so their SD across seeds is zero (the random forest is the exception)."),
  figure("fig1_benchmark.png"),
  caption("Figure 1.", "Balanced accuracy of all baselines with 95% participant-bootstrap confidence intervals, coloured by model family."),
  h2("5.1 Observations"),
  bullet("The differences at the top are small. After Holm correction, the best model is not significantly better than any of the EEG-specific CNNs, ATCNet or EEG Conformer. It is significantly better than MDM, the vanilla RNN, LSTM, DeepConvNet, ViT, Swin and both graph networks (Friedman χ² = 283.0, p < 10⁻⁴⁵)."),
  bullet("Deep models reach their best validation loss after only one to three epochs in several cases (ShallowConvNet, 1D-CNN, ViT, TSception), which suggests they quickly start fitting participant-specific patterns rather than workload."),
  bullet("Decisions made per recording are much more reliable than per window: 0.90 to 0.98 for the leading models."),
  bullet("Feature-based models are well calibrated (ECE about 0.02), while EEG Conformer is noticeably over-confident (ECE 0.135). DeepConvNet was unstable across seeds under the shared recipe."),
  tableCaption("Table 3.", "Cost of selected deep models (one 4-second window; latency measured with batch size 1 on an RTX A4000)."),
  table(["Model", "Parameters", "MFLOPs", "GPU latency (ms)", "Train time per fold (s)"], [
    ["EEGNet-8,2", "1,842", "7.7", "0.23", "10.3"],
    ["EEG-TCNet", "3,814", "2.2", "0.46", "11.5"],
    ["TSception", "7,838", "24.2", "0.29", "18.1"],
    ["ShallowConvNet", "27,682", "29.7", "0.10", "6.8"],
    ["ATCNet", "112,650", "11.8", "2.01", "22.8"],
    ["BiLSTM", "140,546", "141.6", "1.29", "23.2"],
    ["EEG Conformer", "745,506", "44.4", "0.89", "19.2"],
  ], [2600, 1600, 1400, 1700, 1726]),
  p("EEGNet is worth noting here: with fewer than 2,000 parameters it sits within three points of the best model, which matters for real-time use on small devices.", { spacing: { before: 160, after: 120 } }),

  // 6
  h1("6. Neurophysiological check"),
  p("Before trusting any classifier, we checked whether the recordings show the changes that the workload literature predicts. For each participant we averaged log band power per channel in each condition and ran paired t-tests across the 48 participants, with Benjamini–Hochberg correction over the 70 channel-band combinations (Table 4)."),
  tableCaption("Table 4.", "Largest task-minus-rest differences in log band power (paired t-test, n = 48)."),
  table(["Channel", "Band", "t", "Effect size d_z", "q (FDR)"], [
    ["F8", "theta", "+8.61", "+1.24", "2.2 × 10⁻⁹"],
    ["FC6", "theta", "+6.68", "+0.96", "8.8 × 10⁻⁷"],
    ["F8", "delta", "+5.87", "+0.85", "9.9 × 10⁻⁶"],
    ["F8", "gamma", "+5.78", "+0.83", "1.0 × 10⁻⁵"],
    ["T8", "theta", "+5.59", "+0.81", "1.3 × 10⁻⁵"],
    ["AF4", "theta", "+3.93", "+0.57", "1.5 × 10⁻³"],
    ["P8", "alpha", "−3.02", "−0.44", "0.016"],
    ["O2", "alpha", "−2.88", "−0.42", "0.021"],
  ], [1700, 1500, 1500, 2100, 2226]),
  p("Frontal theta rises and posterior alpha falls during the task, which is the classic workload pattern. The same frontal-lateral sites (F7, F8) and T8, however, also rise across every band from delta to gamma. That broadband pattern is typical of eye movements and muscle activity rather than cortical rhythms, which is not surprising for a visual search task such as SIMKAP. This observation motivated the ablation study in the next section.", { spacing: { before: 160, after: 120 } }),

  // 7
  h1("7. How much do artifacts contribute?"),
  p("We repeated the benchmark for nine of the strongest models under three conditions: removing everything above 30 Hz (where most muscle activity sits), zeroing the four eye-adjacent channels F7, F8, T7 and T8, and doing both. The folds and settings were unchanged."),
  tableCaption("Table 5.", "Balanced accuracy with and without the artifact-prone information (3 seeds for the ablations)."),
  table(["Model", "All channels", "No gamma", "No F7/F8/T7/T8", "Both removed"], [
    ["Spectral + SVM", "0.829", "0.825", "0.778", "0.785"],
    ["Spectral + LR", "0.829", "0.824", "0.777", "0.776"],
    ["Spectral + sLDA", "0.827", "0.819", "0.770", "0.769"],
    ["Riemann TS + LR", "0.795", "0.799", "0.776", "0.777"],
    ["ATCNet", "0.803", "0.797", "0.783", "0.777"],
    ["EEGNet-8,2", "0.801", "0.795", "0.780", "0.770"],
    ["EEG-TCNet", "0.797", "0.789", "0.775", "0.769"],
    ["TSception", "0.794", "0.805", "0.780", "0.770"],
    ["ShallowConvNet", "0.785", "0.786", "0.768", "0.759"],
  ], [2600, 1600, 1500, 1700, 1626]),
  figure("fig2_ablation.png"),
  caption("Figure 2.", "Effect of removing high-frequency content and the eye-adjacent channels."),
  p("Removing the gamma band changes accuracy by one point or less, so muscle activity is not the main issue. Zeroing F7, F8, T7 and T8 accounts for almost the whole drop. The spectral models lose between 5.1 and 5.7 points while the deep models lose 1.4 to 2.1, and without these channels ATCNet (0.783), EEGNet and TSception (0.780) move ahead of the feature-based methods. In other words, part of the advantage of hand-crafted features in Table 2 comes from ocular activity at the edges of the headset. Accuracy nevertheless stays near 0.78, far above chance, so most of the decodable information is genuinely neural."),

  // 8
  h1("8. Data augmentation study"),
  p("STEW is balanced between classes, so augmentation here is not about class imbalance. The question is whether extra synthetic or transformed training data helps a model generalise to people it has never seen, particularly when few training participants are available. We compared eight standard EEG augmentations with a conditional Wasserstein GAN (CWGAN-GP) and a conditional diffusion model (DDPM). Both generators were trained separately inside every fold, on training participants only. Three classifiers were chosen in advance (EEGNet, ShallowConvNet and EEG Conformer) and the number of training participants per fold was set to 6, 12, 24 or all of them (about 37)."),
  tableCaption("Table 6.", "Balanced accuracy with all training participants; Δ is the paired change against no augmentation (* Holm-corrected p < 0.05)."),
  table(["Augmentation", "EEG Conformer", "EEGNet-8,2", "ShallowConvNet"], [
    ["None", "0.776", "0.801", "0.794"],
    ["Channel dropout", "0.801 (+0.027*)", "0.807 (+0.007)", "0.810 (+0.017)"],
    ["Mixup", "0.798 (+0.023*)", "0.795 (−0.006)", "0.797 (+0.004)"],
    ["Gaussian noise", "0.792 (+0.017*)", "0.801 (+0.001)", "0.793 (−0.000)"],
    ["CWGAN-GP, 50% extra", "0.787 (+0.012)", "0.778 (−0.021*)", "0.779 (−0.015)"],
    ["CWGAN-GP, 100% extra", "0.782 (+0.008)", "0.768 (−0.033*)", "0.769 (−0.025*)"],
    ["DDPM, 50% extra", "0.770 (−0.004)", "0.804 (+0.004)", "0.793 (+0.000)"],
    ["DDPM, 100% extra", "0.780 (+0.005)", "0.790 (−0.010)", "0.799 (+0.007)"],
  ], [2600, 2150, 2150, 2126]),
  figure("fig3_data_efficiency.png"),
  caption("Figure 3.", "Balanced accuracy as the number of training participants grows."),
  p("The picture depends on the model and on how much real data is available. With 12 training participants, synthetic data clearly helps the transformer: EEG Conformer rises from 0.668 to 0.752 with the GAN (+8.2 points) and to 0.733 with diffusion (+6.4 points), and diffusion also helps ShallowConvNet (+3.0). With all participants the GAN reduces the accuracy of EEGNet and ShallowConvNet, the diffusion model has no significant effect, and simple channel dropout is the most consistent improvement across all three networks."),
  tableCaption("Table 7.", "Quality of the synthetic data (averaged over folds, seeds and classes). LSD = log-spectral distance; TSTR = trained on synthetic, tested on real."),
  table(["Training set", "Generator", "LSD (dB)", "MMD", "Coverage", "TSTR BAcc", "TRTR BAcc"], [
    ["12 participants", "CWGAN-GP", "2.71", "0.046", "0.365", "0.582", "0.713"],
    ["12 participants", "DDPM", "3.65", "0.141", "0.367", "0.688", "0.713"],
    ["All participants", "CWGAN-GP", "2.55", "0.034", "0.572", "0.600", "0.760"],
    ["All participants", "DDPM", "5.12", "0.246", "0.453", "0.707", "0.760"],
  ], [1700, 1300, 1100, 1100, 1250, 1300, 1276]),
  p("One interesting point is that the GAN produces signals whose spectra and covariance are closer to the real data (lower LSD and MMD), yet a classifier trained only on its samples reaches 0.60, compared with 0.71 for the diffusion samples. Realistic-looking signals are not necessarily informative about the class, which is consistent with the GAN harming the compact CNNs when plenty of real data is available.", { spacing: { before: 160, after: 120 } }),

  // 9
  h1("9. Proposed method: artifact-counterfactual consistency training"),
  h2("9.1 Motivation"),
  p("Sections 6 and 7 show that a classifier can score well on STEW partly by reading eye activity. This is a real problem for deployment: a person looking around more, or a VR headset pressing on the forehead electrodes, could change the workload estimate without any change in mental effort. Consumer headsets have no EOG channel that could be used to remove these signals. The goal of the proposed method is to train a classifier whose decision does not change when artifacts are added or removed, without needing any extra sensor."),
  h2("9.2 Architecture"),
  figure("fig4_model_e.png"),
  caption("Figure 4.", "Overview of the proposed training scheme. The encoder is unchanged at inference time; only the training procedure differs."),
  p("The method has four parts."),
  numbered([t("Artifact bank. ", { bold: true }), t("For each fold, we estimate the spatial patterns of ocular activity from the training participants only: the principal components of the slow (below 4 Hz) activity that load mainly on the frontal electrodes, together with a library of their real time courses. Muscle activity is described by the channels that show excess 20–45 Hz power. Both classes are pooled, so the bank carries no information about the label.")]),
  numbered([t("Counterfactual generator. ", { bold: true }), t("Each training window receives a partner copy in which ocular waveforms are added along the estimated patterns, the ocular subspace is removed from the slow activity, or band-limited muscle bursts are added. The choice of operation is random and independent of the class, and every window is re-standardised afterwards.")]),
  numbered([t("Consistency objective. ", { bold: true }), t("The original window x and its counterfactual x̃ pass through the same encoder. The loss combines cross-entropy on both with a symmetric Kullback–Leibler term that penalises any difference between the two predictions:")]),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
    children: [t("L = ½ CE(x, y) + ½ CE(x̃, y) + λ · [ KL(p(x) ‖ p(x̃)) + KL(p(x̃) ‖ p(x)) ] / 2,   λ = 3", { italics: true, size: 21 })] }),
  numbered([t("Channel dropout. ", { bold: true }), t("Whole channels are randomly zeroed during training (probability 0.5 per window), which the augmentation study identified as the most reliable simple augmentation and which complements the counterfactuals.")]),
  p("The encoder itself can be any network. EEGNet was used for development because it is small, fast and stable. A differentiable version of the spectral-feature pipeline (SpecNet, about 7,000 parameters) has also been implemented, so that the most accurate model family can be trained with the same objective."),
  h2("9.3 Development protocol"),
  p("To avoid tuning on test data, every design decision so far was made in a development mode that evaluates only on the validation participants of each fold; the test participants have not been used. Besides clean accuracy, three robustness measures are recorded: accuracy when F7, F8, T7 and T8 are zeroed at test time, accuracy and the share of flipped predictions when artifacts are injected into the test windows, and the accuracy gap between each participant's most and least eye-active windows (which involves no synthetic artifacts at all)."),
  h2("9.4 Development results"),
  tableCaption("Table 8.", "Development results with EEGNet (validation participants only, 3 seeds). CD = channel dropout."),
  table(["Training scheme", "Clean BAcc", "AUC", "ECE", "BAcc, F7/F8/T7/T8 zeroed", "Flips under artifacts ×3"], [
    ["Standard training", "0.768 ± 0.007", "0.836", "0.064", "0.630", "24.5%"],
    ["Channel dropout only", "0.778 ± 0.020", "0.849", "0.058", "0.683", "16.7%"],
    ["Counterfactuals, no consistency term", "0.768 ± 0.007", "0.839", "0.061", "0.635", "15.1%"],
    ["ACCT, λ = 1", "0.780 ± 0.012", "0.854", "0.056", "0.663", "13.4%"],
    ["ACCT, λ = 3", "0.790 ± 0.011", "0.860", "0.046", "0.673", "10.6%"],
    ["ACCT + CD, λ = 1", "0.792 ± 0.002", "0.871", "0.032", "0.711", "7.8%"],
    ["ACCT + CD, λ = 3", "0.807 ± 0.008", "0.880", "0.030", "0.710", "7.0%"],
  ], [2700, 1300, 900, 900, 1700, 1526], { highlightRows: [6] }),
  figure("fig5_model_e_dev.png"),
  caption("Figure 5.", "Development results: clean and channel-ablated accuracy (left) and prediction flips under injected artifacts (right)."),
  p("Compared with standard training, the full scheme raises clean balanced accuracy by 3.8 points (95% CI +1.7 to +6.0; the only clean improvement that stays significant after Holm correction, p = 0.035), increases AUC from 0.836 to 0.880, halves the calibration error and cuts flipped predictions under strong artifacts from 24.5% to 7.0%. Accuracy with the four eye-adjacent channels removed improves by 7.9 points. The row without the consistency term shows almost no gain on clean data, which indicates that the consistency objective, rather than the extra training windows, is what drives the improvement."),
  p("Two limitations should be stated clearly. First, the injected-artifact test uses the same family of artifacts that the method is trained on, so it naturally favours the method. Second, on the more neutral test that compares each participant's most and least eye-active windows, the gap (about 10 points) is the same for all training schemes. The method raises accuracy on both kinds of window rather than closing the gap, so those windows may simply be noisier. These results also come from validation participants only and still need confirmation on the held-out test participants."),

  // 10
  h1("10. Current work and next steps"),
  bullet([t("Development round 3 (running). ", { bold: true }), t("Applying the scheme to ATCNet, TSception, EEG-TCNet and SpecNet, to check whether the gains carry over to other architectures.")]),
  bullet([t("Final evaluation. ", { bold: true }), t("Once the design is fixed, a single run on the held-out test participants with five seeds, compared against the baselines in Table 2. The design will not be changed after this run.")]),
  bullet([t("Second dataset. ", { bold: true }), t("Confirmation on an independent workload dataset (for example COG-BCI), restricted to the 14 Emotiv-equivalent channels at 128 Hz.")]),
  bullet([t("Own recordings. ", { bold: true }), t("An Emotiv EPOC headset is expected. The plan is a small study, after ethics approval, that combines it with a Meta Quest 3S: controlled workload tasks inside VR with graded difficulty, and a short cued calibration (look left and right, blink, clench) to obtain labelled artifact examples for each person.")]),
  bullet([t("Intellectual property. ", { bold: true }), t("A short invention disclosure will be prepared for the VIT IPR cell before any public presentation, preprint or open release of the code.")]),
  bullet([t("Paper. ", { bold: true }), t("A draft covering the benchmark, the artifact analysis and the augmentation study is in progress; the proposed method will be added after the final evaluation.")]),

  // 11
  h1("11. Implementation notes"),
  p("All experiments run inside Docker on the lab workstation (NVIDIA RTX A4000, 16 GB) using the NVIDIA NGC PyTorch 26.07 container, in line with the lab's policy. The code is organised as a Python package with configuration files for every experiment. Each run stores the code revision, package versions and a checksum of the raw data, so results can be traced and reproduced. Every run can be resumed after an interruption, and an automated test suite checks, among other things, that no participant ever appears in more than one data split."),

  // refs
  h1("References"),
  ...[
    "W. L. Lim, O. Sourina and L. P. Wang, \"STEW: Simultaneous task EEG workload data set,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 26, no. 11, pp. 2106–2114, 2018.",
    "V. J. Lawhern et al., \"EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,\" J. Neural Eng., vol. 15, no. 5, 056013, 2018.",
    "R. T. Schirrmeister et al., \"Deep learning with convolutional neural networks for EEG decoding and visualization,\" Hum. Brain Mapp., vol. 38, no. 11, pp. 5391–5420, 2017.",
    "T. M. Ingolfsson et al., \"EEG-TCNet: an accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces,\" Proc. IEEE SMC, 2020.",
    "Y. Ding et al., \"TSception: capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition,\" IEEE Trans. Affective Comput., 2023.",
    "Y. Song et al., \"EEG Conformer: convolutional transformer for EEG decoding and visualization,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 31, pp. 710–719, 2023.",
    "H. Altaheri, G. Muhammad and M. Alsulaiman, \"Physics-informed attention temporal convolutional network for EEG-based motor imagery classification,\" IEEE Trans. Ind. Informat., vol. 19, no. 2, pp. 2249–2258, 2023.",
    "T. Song, W. Zheng, P. Song and Z. Cui, \"EEG emotion recognition using dynamical graph convolutional neural networks,\" IEEE Trans. Affective Comput., 2018.",
    "A. Barachant et al., \"Multiclass brain–computer interface classification by Riemannian geometry,\" IEEE Trans. Biomed. Eng., vol. 59, no. 4, pp. 920–928, 2012.",
    "P. Zanini et al., \"Transfer learning: a Riemannian geometry framework with applications to brain–computer interfaces,\" IEEE Trans. Biomed. Eng., vol. 65, no. 5, 2018.",
    "I. Gulrajani et al., \"Improved training of Wasserstein GANs,\" Proc. NeurIPS, 2017.",
    "J. Ho, A. Jain and P. Abbeel, \"Denoising diffusion probabilistic models,\" Proc. NeurIPS, 2020.",
    "C. Rommel et al., \"Data augmentation for learning predictive models on EEG: a systematic comparison,\" J. Neural Eng., vol. 19, no. 6, 066020, 2022.",
    "J. Demšar, \"Statistical comparisons of classifiers over multiple data sets,\" J. Mach. Learn. Res., vol. 7, pp. 1–30, 2006.",
    "W. Klimesch, \"EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis,\" Brain Res. Rev., vol. 29, pp. 169–195, 1999.",
  ].map((r, i) => new Paragraph({ spacing: { after: 80 }, indent: { left: 440, hanging: 440 },
      children: [t(`[${i + 1}]  `, { size: 19 }), t(r, { size: 19 })] })),
];

const doc = new Document({
  creator: "Aryan Kalra",
  title: "Progress report: subject-independent EEG workload classification on STEW",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2F5680" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
                    titlePage: true },
      headers: {
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "B8C4D2", space: 4 } },
          children: [t("EEG workload classification on STEW · Progress report · Confidential", { size: 16, color: "6B7C93" })] })] }),
        first: new Header({ children: [new Paragraph({ children: [] })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [t("Page ", { size: 17, color: "6B7C93" }), new TextRun({ children: [PageNumber.CURRENT], size: 17, font: FONT, color: "6B7C93" })] })] }),
        first: new Footer({ children: [new Paragraph({ children: [] })] }),
      },
      children: [...titlePage, ...toc, ...body] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(DIR, "STEW_Progress_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
