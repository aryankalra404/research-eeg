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
      altText: { title: "Figure", description: "Figure", name: "Figure" } })] });
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

// ---------- compact results document ----------
const kv = (rows, widths = [2600, 6426]) => table(["Item", "Setting"], rows, widths, { boldFirstCol: true, fontSize: 17 });
const b = (label, text) => bullet([t(label + " ", { bold: true }), t(text)]);
const gap = () => new Paragraph({ spacing: { after: 60 }, children: [] });

const body = [
  new Paragraph({ spacing: { after: 60 }, children: [t("Subject-Independent EEG Mental-Workload Classification on STEW", { size: 34, bold: true, color: ACCENT })] }),
  new Paragraph({ spacing: { after: 240 }, children: [t("Baselines · artifact analysis · augmentation · proposed artifact-robust training (ACCT)", { size: 20, italics: true, color: "555555" })] }),

  // ---- key numbers
  tableCaption("Table 1.", "Key numbers at a glance."),
  table(["Result", "Value"], [
    ["Best baseline (Spectral + LR), balanced accuracy", "0.829  [0.796, 0.861]"],
    ["Best deep baseline (ATCNet)", "0.803 ± 0.010"],
    ["Best model without F7/F8/T7/T8 (ATCNet)", "0.783"],
    ["Largest augmentation gain (EEG Conformer, 12 subjects, CWGAN-GP)", "+8.2 points"],
    ["ACCT vs. standard training, EEGNet (validation subjects)", "0.807 vs. 0.768  (+3.8, p_Holm = 0.035)"],
    ["Prediction flips under injected artifacts ×3", "24.5% → 7.0%"],
    ["Calibration error (ECE)", "0.064 → 0.030"],
  ], [6026, 3000], { fontSize: 17 }),

  // ---- 1 setup
  h1("1. Data and protocol"),
  kv([
    ["Dataset", "STEW: 48 subjects, Emotiv EPOC, 14 channels, 128 Hz; rest vs. SIMKAP multitasking, 150 s each"],
    ["Channels", "AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4"],
    ["Filtering", "0.5–45 Hz, 4th-order Butterworth, zero-phase"],
    ["Windows", "4 s (512 samples), 50% overlap"],
    ["Artifact rejection", "Peak-to-peak > median + 60 × MAD, per subject, class-agnostic"],
    ["Normalisation", "z-score per window and channel (no per-recording normalisation: would leak the label)"],
    ["Split", "10-fold cross-validation by subject; each subject tested once; 5 seeds"],
    ["Validation", "15% of training subjects (subject-disjoint), early stopping only"],
    ["Deep training", "AdamW, lr 1e-3, batch 64, cosine schedule + warm-up, ≤ 100 epochs, same recipe for all"],
    ["Metrics", "Balanced accuracy (primary), accuracy, macro-F1, κ, MCC, ROC-AUC, PR-AUC, sensitivity, specificity, Brier, log-loss, ECE; window and recording level"],
    ["Statistics", "Subject-cluster bootstrap 95% CI; Friedman + Nemenyi; Wilcoxon signed-rank + Holm"],
    ["Hardware", "NVIDIA RTX A4000 16 GB, NGC PyTorch 26.07 container"],
  ]),

  // ---- 2 benchmark
  h1("2. Baseline benchmark (25 models)"),
  tableCaption("Table 2.", "Subject-independent results, mean ± SD over 5 seeds. † uses unlabelled test-subject data (transductive)."),
  table(["Model", "Family", "BAcc", "95% CI", "AUC", "κ", "ECE", "Rec. Acc"], benchRows,
        [2350, 1450, 1050, 1250, 700, 700, 700, 826], { fontSize: 15, highlightRows: [0] }),
  note("Classical models are deterministic given the folds (SD = 0), except RF. Rec. Acc = accuracy after averaging each 150 s recording."),
  figure("fig1_benchmark.png"),
  caption("Figure 1.", "Balanced accuracy with 95% subject-bootstrap CIs, by model family."),
  b("Friedman:", "χ² = 283.0, p = 4.3 × 10⁻⁴⁶; Nemenyi CD = 5.50 ranks."),
  b("Wilcoxon vs. best (Holm):", "significantly worse only for MDM, vanilla RNN, LSTM, DeepConvNet, ViT, Swin, electrode GCN, DGCNN; top EEG CNNs and ATCNet not significantly different."),
  b("Overfitting:", "several deep models peak at validation epoch 1–3 (ShallowConvNet, 1D-CNN, ViT, TSception)."),
  b("Calibration:", "feature models ECE ≈ 0.02; EEG Conformer over-confident (0.135); DeepConvNet unstable across seeds."),
  tableCaption("Table 3.", "Model cost (one 4 s window; GPU latency at batch size 1)."),
  table(["Model", "Parameters", "MFLOPs", "GPU latency (ms)", "Train time / fold (s)"], [
    ["EEGNet-8,2", "1,842", "7.7", "0.23", "10.3"],
    ["EEG-TCNet", "3,814", "2.2", "0.46", "11.5"],
    ["TSception", "7,838", "24.2", "0.29", "18.1"],
    ["ShallowConvNet", "27,682", "29.7", "0.10", "6.8"],
    ["ATCNet", "112,650", "11.8", "2.01", "22.8"],
    ["BiLSTM", "140,546", "141.6", "1.29", "23.2"],
    ["EEG Conformer", "745,506", "44.4", "0.89", "19.2"],
  ], [2600, 1600, 1400, 1700, 1726]),

  // ---- 3 neuro
  h1("3. Task vs. rest band power"),
  tableCaption("Table 4.", "Largest task − rest differences in log band power (paired t, n = 48, BH-FDR over 70 tests)."),
  table(["Channel", "Band", "t", "d_z", "q (FDR)"], [
    ["F8", "theta", "+8.61", "+1.24", "2.2 × 10⁻⁹"],
    ["FC6", "theta", "+6.68", "+0.96", "8.8 × 10⁻⁷"],
    ["F8", "delta", "+5.87", "+0.85", "9.9 × 10⁻⁶"],
    ["F8", "gamma", "+5.78", "+0.83", "1.0 × 10⁻⁵"],
    ["T8", "theta", "+5.59", "+0.81", "1.3 × 10⁻⁵"],
    ["AF4", "theta", "+3.93", "+0.57", "1.5 × 10⁻³"],
    ["P8", "alpha", "−3.02", "−0.44", "0.016"],
    ["O2", "alpha", "−2.88", "−0.42", "0.021"],
  ], [1700, 1500, 1500, 2100, 2226]),
  b("Expected workload pattern:", "frontal theta ↑, posterior alpha ↓."),
  b("Artifact sign:", "F7, F8, T8 rise in every band (delta–gamma), typical of eye/muscle activity."),

  // ---- 4 ablation
  h1("4. Artifact ablation"),
  tableCaption("Table 5.", "Balanced accuracy when artifact-prone information is removed (3 seeds, same folds)."),
  table(["Model", "All channels", "No gamma", "No F7/F8/T7/T8", "Both removed", "Δ channels"], [
    ["Spectral + SVM", "0.829", "0.825", "0.778", "0.785", "−5.1"],
    ["Spectral + LR", "0.829", "0.824", "0.777", "0.776", "−5.2"],
    ["Spectral + sLDA", "0.827", "0.819", "0.770", "0.769", "−5.7"],
    ["Riemann TS + LR", "0.795", "0.799", "0.776", "0.777", "−1.9"],
    ["ATCNet", "0.803", "0.797", "0.783", "0.777", "−2.0"],
    ["EEGNet-8,2", "0.801", "0.795", "0.780", "0.770", "−2.1"],
    ["EEG-TCNet", "0.797", "0.789", "0.775", "0.769", "−2.2"],
    ["TSception", "0.794", "0.805", "0.780", "0.770", "−1.4"],
    ["ShallowConvNet", "0.785", "0.786", "0.768", "0.759", "−1.7"],
  ], [2300, 1400, 1300, 1600, 1400, 1026]),
  figure("fig2_ablation.png"),
  caption("Figure 2.", "Effect of removing gamma and the eye-adjacent channels."),
  b("Gamma:", "≤ 1 point change → muscle activity is not the main factor."),
  b("F7/F8/T7/T8:", "spectral models lose 5–6 points, deep models 1.4–2.2; ranking flips to deep models (ATCNet 0.783)."),
  b("Still ~0.78 without them:", "most of the decodable signal is neural."),

  // ---- 5 augmentation
  h1("5. Augmentation: standard vs. GAN vs. diffusion"),
  note("8 online augmentations, CWGAN-GP and conditional DDPM; generators trained inside each fold on training subjects only; 3 seeds."),
  tableCaption("Table 6.", "All training subjects. Δ = paired change vs. none (* Holm p < 0.05)."),
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
  tableCaption("Table 7.", "Balanced accuracy by number of training subjects per fold (6 / 12 / 24 / all)."),
  table(["Augmentation", "EEG Conformer", "EEGNet-8,2", "ShallowConvNet"], [
    ["None", ".632 / .668 / .773 / .776", ".712 / .742 / .775 / .801", ".733 / .743 / .774 / .794"],
    ["CWGAN-GP 100%", ".661 / .752 / .764 / .782", ".710 / .727 / .738 / .768", ".735 / .756 / .777 / .769"],
    ["DDPM 100%", ".641 / .733 / .773 / .780", ".703 / .727 / .763 / .790", ".726 / .772 / .780 / .799"],
    ["Mixup", ".679 / .698 / .780 / .798", ".707 / .727 / .770 / .795", ".710 / .746 / .775 / .797"],
    ["Channel dropout", ".657 / .674 / .778 / .801", ".717 / .745 / .780 / .807", ".745 / .752 / .788 / .810"],
    ["Frequency shift", ".698 / .686 / .762 / .789", ".732 / .745 / .773 / .800", ".714 / .738 / .772 / .788"],
  ], [1900, 2400, 2350, 2376], { fontSize: 15 }),
  figure("fig3_data_efficiency.png"),
  caption("Figure 3.", "Accuracy vs. training subjects per fold."),
  tableCaption("Table 8.", "Synthetic-data quality (mean over folds, seeds, classes). TSTR = train synthetic, test real; TRTR = real → real."),
  table(["Subjects", "Generator", "LSD (dB)", "MMD", "Density", "Coverage", "Memor.", "TSTR", "TRTR"], [
    ["6", "CWGAN-GP", "2.79", "0.074", "0.648", "0.325", "0.981", "0.603", "0.690"],
    ["6", "DDPM", "3.92", "0.321", "0.446", "0.138", "0.989", "0.647", "0.690"],
    ["12", "CWGAN-GP", "2.71", "0.046", "0.732", "0.365", "0.952", "0.582", "0.713"],
    ["12", "DDPM", "3.65", "0.141", "1.081", "0.367", "0.931", "0.688", "0.713"],
    ["24", "CWGAN-GP", "2.85", "0.037", "1.068", "0.488", "0.935", "0.599", "0.747"],
    ["24", "DDPM", "4.67", "0.211", "0.985", "0.403", "0.915", "0.709", "0.747"],
    ["all", "CWGAN-GP", "2.55", "0.034", "1.278", "0.572", "0.941", "0.600", "0.760"],
    ["all", "DDPM", "5.12", "0.246", "0.997", "0.453", "0.907", "0.707", "0.760"],
  ], [950, 1250, 1000, 950, 1000, 1050, 950, 950, 926], { fontSize: 15 }),
  b("Few subjects (12):", "synthetic data helps the transformer: Conformer +8.2 (CWGAN), +6.4 (DDPM); ShallowConvNet + DDPM +3.0."),
  b("All subjects:", "CWGAN hurts EEGNet (−3.3) and ShallowConvNet (−2.5); DDPM neutral; channel dropout most consistent."),
  b("Fidelity ≠ usefulness:", "CWGAN matches spectra better (lower LSD/MMD) but carries less class information (TSTR 0.60 vs. 0.71)."),

  // ---- 6 ACCT
  h1("6. Proposed method: ACCT"),
  note("Artifact-counterfactual consistency training: the classifier is trained to give the same prediction for a window and for a copy with eye/muscle artifacts added or removed. No extra sensor (EOG) needed; inference is unchanged."),
  figure("fig4_model_e.png"),
  caption("Figure 4.", "ACCT training scheme."),
  tableCaption("Table 9.", "Components."),
  table(["Component", "What it does"], [
    ["Artifact bank", "Per fold, training subjects only, both classes pooled. Ocular: frontal-dominant principal components of < 4 Hz activity + bank of real waveforms. EMG: channel profile of excess 20–45 Hz power."],
    ["Counterfactual generator", "Per window, label-independent: add ocular (p = 0.5), remove ocular subspace (p = 0.3), add EMG bursts (p = 0.5); re-z-scored."],
    ["Consistency loss", "L = ½ CE(x, y) + ½ CE(x̃, y) + λ · ½ [KL(p(x)‖p(x̃)) + KL(p(x̃)‖p(x))],  λ = 3"],
    ["Channel dropout", "Whole channels zeroed at random during training."],
    ["Encoder", "Any network. EEGNet for development; ATCNet, TSception, EEG-TCNet and SpecNet (differentiable spectral model, ~7k parameters) in round 3."],
  ], [2400, 6626], { boldFirstCol: true, fontSize: 16 }),
  tableCaption("Table 10.", "Development results, EEGNet, validation subjects only (3 seeds). CD = channel dropout."),
  table(["Training", "Clean BAcc", "AUC", "ECE", "BAcc F7/F8/T7/T8 = 0", "BAcc high-ocular", "Natural gap"], [
    ["Standard (ERM)", "0.768 ± 0.007", "0.836", "0.064", "0.630", "0.698", "+0.107"],
    ["Channel dropout", "0.778 ± 0.020", "0.849", "0.058", "0.683", "0.710", "+0.105"],
    ["Counterfactuals only", "0.768 ± 0.007", "0.839", "0.061", "0.635", "—", "—"],
    ["ACCT λ = 1", "0.780 ± 0.012", "0.854", "0.056", "0.663", "0.716", "+0.099"],
    ["ACCT λ = 3", "0.790 ± 0.011", "0.860", "0.046", "0.673", "0.726", "+0.099"],
    ["ACCT + CD λ = 1", "0.792 ± 0.002", "0.871", "0.032", "0.711", "0.718", "+0.120"],
    ["ACCT + CD λ = 3", "0.807 ± 0.008", "0.880", "0.030", "0.710", "0.730", "+0.117"],
  ], [2100, 1350, 850, 850, 1600, 1400, 876], { fontSize: 15, highlightRows: [6] }),
  tableCaption("Table 11.", "Injected-artifact stress test: balanced accuracy / predictions flipped."),
  table(["Training", "Strength ×1", "Strength ×2", "Strength ×3"], [
    ["Standard (ERM)", "0.752 / 7.6%", "0.707 / 17.5%", "0.667 / 24.5%"],
    ["Channel dropout", "0.771 / 4.8%", "0.748 / 11.0%", "0.723 / 16.7%"],
    ["Counterfactuals only", "0.766 / 4.2%", "0.752 / 9.7%", "0.732 / 15.1%"],
    ["ACCT λ = 1", "0.776 / 3.7%", "0.765 / 8.2%", "0.744 / 13.4%"],
    ["ACCT λ = 3", "—", "—", "0.761 / 10.6%"],
    ["ACCT + CD λ = 1", "—", "—", "0.778 / 7.8%"],
    ["ACCT + CD λ = 3", "—", "—", "0.793 / 7.0%"],
  ], [2700, 2100, 2100, 2126], { highlightRows: [6] }),
  figure("fig5_model_e_dev.png"),
  caption("Figure 5.", "Clean vs. channel-ablated accuracy (left); flips under injected artifacts (right)."),
  tableCaption("Table 12.", "Paired per-subject tests vs. standard training (Wilcoxon, Holm)."),
  table(["Training", "Metric", "Δ", "p_Holm"], [
    ["ACCT + CD λ = 3", "Clean BAcc", "+0.038 [+0.017, +0.060]", "0.035"],
    ["ACCT + CD λ = 3", "BAcc, F7/F8/T7/T8 zeroed", "+0.079", "0.00059"],
    ["ACCT + CD λ = 1", "BAcc, F7/F8/T7/T8 zeroed", "+0.081", "1.1 × 10⁻⁶"],
    ["ACCT λ = 3", "BAcc, F7/F8/T7/T8 zeroed", "+0.043", "0.0025"],
    ["ACCT λ = 1", "Clean BAcc", "+0.012", "0.14 (n.s.)"],
    ["Counterfactuals only", "Clean BAcc", "≈ 0", "n.s."],
  ], [2400, 2600, 2400, 1626]),
  b("Active ingredient:", "the consistency term (counterfactuals alone give no clean gain)."),
  b("Best configuration:", "ACCT + channel dropout, λ = 3."),
  b("Limitations:", "stress test uses the same artifact family as training; natural gap (most vs. least eye-active windows) unchanged at ≈ 0.10–0.12; validation subjects only, test run pending."),

  // ---- 7 status
  h1("7. Status"),
  table(["Item", "Status"], [
    ["25-model benchmark, neurophysiology, ablations, augmentation study", "Done"],
    ["ACCT development rounds 1–2 (EEGNet)", "Done"],
    ["Round 3: ACCT on ATCNet, TSception, EEG-TCNet, SpecNet", "Running"],
    ["Final held-out test (5 seeds, run once after design freeze)", "Pending"],
    ["Confirmation on a second dataset (e.g. COG-BCI)", "Planned"],
    ["Own recordings: Emotiv EPOC + Meta Quest 3S (after ethics approval)", "Planned"],
  ], [6626, 2400], { fontSize: 17 }),
];

const doc = new Document({
  creator: "", title: "STEW workload classification results",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: "2F5680" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 400, hanging: 220 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 17, font: FONT, color: "6B7C93" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(DIR, "STEW_Results.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
