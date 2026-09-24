"""Model registry: every baseline with its family, input and reference."""

from __future__ import annotations

from dataclasses import dataclass

from .. import constants as C


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display: str
    kind: str          # "deep" | "classical"
    family: str
    input: str
    reference: str
    transductive: bool = False  # uses unlabelled test-subject data
    notes: str = ""


def _deep(module, cls):
    def build(n_channels=C.N_CHANNELS, n_times=512, n_classes=2, sfreq=C.SFREQ, **kwargs):
        import importlib
        model_cls = getattr(importlib.import_module(f"stewbench.models.{module}"), cls)
        return model_cls(n_channels, n_times, n_classes, sfreq, **kwargs)
    return build


SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    # --- classical ---------------------------------------------------------
    ModelSpec("bp_lda", "Spectral + sLDA", "classical", "Feature-based", "spectral features",
              "Lotte et al., J. Neural Eng. 2018"),
    ModelSpec("bp_lr", "Spectral + LR", "classical", "Feature-based", "spectral features",
              "Lotte et al., J. Neural Eng. 2018"),
    ModelSpec("bp_svm", "Spectral + SVM-RBF", "classical", "Feature-based", "spectral features",
              "Lotte et al., J. Neural Eng. 2018"),
    ModelSpec("bp_rf", "Spectral + RF", "classical", "Feature-based", "spectral features",
              "Breiman, Mach. Learn. 2001"),
    ModelSpec("bp_gbdt", "Spectral + GBDT", "classical", "Feature-based", "spectral features",
              "Ke et al., NeurIPS 2017 (histogram GBDT)"),
    ModelSpec("riemann_mdm", "Riemann MDM", "classical", "Riemannian", "spatial covariance",
              "Barachant et al., IEEE TBME 2012"),
    ModelSpec("riemann_ts_lr", "Riemann TS + LR", "classical", "Riemannian", "spatial covariance",
              "Barachant et al., Neurocomputing 2013"),
    ModelSpec("riemann_rpa_ts_lr", "Riemann re-centred TS + LR", "classical", "Riemannian",
              "spatial covariance", "Zanini et al., IEEE TBME 2018", transductive=True,
              notes="re-centres each subject's covariances with its own unlabelled Riemannian mean"),
    # --- generic deep ------------------------------------------------------
    ModelSpec("cnn1d", "1D-CNN", "deep", "Generic CNN", "raw EEG", "LeCun et al., 1998 (generic)"),
    ModelSpec("rnn", "Vanilla RNN", "deep", "Recurrent", "raw EEG sequence", "Elman, Cogn. Sci. 1990"),
    ModelSpec("lstm", "LSTM", "deep", "Recurrent", "raw EEG sequence", "Hochreiter & Schmidhuber, 1997"),
    ModelSpec("bilstm", "BiLSTM", "deep", "Recurrent", "raw EEG sequence", "Schuster & Paliwal, 1997"),
    ModelSpec("gru", "GRU", "deep", "Recurrent", "raw EEG sequence", "Cho et al., 2014"),
    ModelSpec("cnn_lstm", "CNN-LSTM", "deep", "Generic CNN", "raw EEG", "common STEW hybrid baseline"),
    # --- EEG-specific deep -------------------------------------------------
    ModelSpec("eegnet", "EEGNet-8,2", "deep", "EEG CNN", "raw EEG", "Lawhern et al., J. Neural Eng. 2018"),
    ModelSpec("shallowconvnet", "ShallowConvNet", "deep", "EEG CNN", "raw EEG",
              "Schirrmeister et al., Hum. Brain Mapp. 2017", notes="kernels rescaled 250->128 Hz"),
    ModelSpec("deepconvnet", "DeepConvNet", "deep", "EEG CNN", "raw EEG",
              "Schirrmeister et al., Hum. Brain Mapp. 2017", notes="kernels rescaled 250->128 Hz"),
    ModelSpec("eegtcnet", "EEG-TCNet", "deep", "EEG CNN", "raw EEG", "Ingolfsson et al., IEEE SMC 2020",
              notes="kernels rescaled 250->128 Hz"),
    ModelSpec("tsception", "TSception", "deep", "EEG CNN", "raw EEG",
              "Ding et al., IEEE Trans. Affect. Comput. 2023"),
    ModelSpec("eeg_conformer", "EEG Conformer", "deep", "EEG Transformer", "raw EEG",
              "Song et al., IEEE TNSRE 2023", notes="kernels rescaled 250->128 Hz"),
    ModelSpec("atcnet", "ATCNet", "deep", "EEG Transformer", "raw EEG",
              "Altaheri et al., IEEE Trans. Ind. Inform. 2023", notes="kernels rescaled 250->128 Hz"),
    ModelSpec("vit_stft", "ViT (STFT)", "deep", "Vision Transformer", "internal log-STFT",
              "Dosovitskiy et al., ICLR 2021", notes="compact EEG adaptation"),
    ModelSpec("swin_stft", "Swin (STFT)", "deep", "Vision Transformer", "internal log-STFT",
              "Liu et al., ICCV 2021", notes="compact EEG adaptation"),
    ModelSpec("gcn", "Electrode GCN", "deep", "Graph", "raw EEG + electrode kNN graph",
              "Kipf & Welling, ICLR 2017"),
    ModelSpec("dgcnn", "DGCNN", "deep", "Graph", "internal band power + learnable graph",
              "Song et al., IEEE Trans. Affect. Comput. 2018"),
]}

DEEP_BUILDERS = {
    "cnn1d": _deep("convolutional", "CNN1D"),
    "rnn": _deep("recurrent", "VanillaRNN"),
    "lstm": _deep("recurrent", "LSTM"),
    "bilstm": _deep("recurrent", "BiLSTM"),
    "gru": _deep("recurrent", "GRU"),
    "cnn_lstm": _deep("convolutional", "CNNLSTM"),
    "eegnet": _deep("convolutional", "EEGNet"),
    "shallowconvnet": _deep("convolutional", "ShallowConvNet"),
    "deepconvnet": _deep("convolutional", "DeepConvNet"),
    "eegtcnet": _deep("convolutional", "EEGTCNet"),
    "tsception": _deep("convolutional", "TSception"),
    "eeg_conformer": _deep("attention", "EEGConformer"),
    "atcnet": _deep("attention", "ATCNet"),
    "vit_stft": _deep("attention", "ViTSTFT"),
    "swin_stft": _deep("attention", "SwinSTFT"),
    "gcn": _deep("graph", "ElectrodeGCN"),
    "dgcnn": _deep("graph", "DGCNN"),
}

CLASSICAL_MODELS = tuple(n for n, s in SPECS.items() if s.kind == "classical")
DEEP_MODELS = tuple(n for n, s in SPECS.items() if s.kind == "deep")
ALL_MODELS = CLASSICAL_MODELS + DEEP_MODELS


def spec(name: str) -> ModelSpec:
    if name not in SPECS:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(SPECS)}")
    return SPECS[name]


def build_deep(name: str, **kwargs):
    if name not in DEEP_BUILDERS:
        raise KeyError(f"{name!r} is not a deep model")
    return DEEP_BUILDERS[name](**kwargs)


def build_classical(name: str, seed: int = 0):
    from .classical import build_classical as _build
    return _build(name, seed)
