import unittest

import numpy as np
import torch

from src import config
from src.models import MODEL_REGISTRY, get_model
from src.evaluation import evaluate_predictions, paired_sign_flip_test
from src.preprocessing import zscore_normalize_per_window
from src.split import inner_group_split_indices
from src.synthetic_quality import evaluate_synthetic_quality
from src.train_gan import ADAM_BETAS
from src.reproducibility import set_seed


class ResearchProtocolTests(unittest.TestCase):
    def test_inner_split_has_no_subject_overlap(self):
        groups = np.repeat(np.arange(1, 11), 4)
        inner_train, inner_val = inner_group_split_indices(groups, seed=42)
        self.assertIsNotNone(inner_val)
        self.assertTrue(set(groups[inner_train]).isdisjoint(set(groups[inner_val])))

    def test_stew_normalization_is_window_local(self):
        rng = np.random.default_rng(42)
        windows = rng.normal(size=(3, 512, 14)).astype(np.float32)
        normalized_before = zscore_normalize_per_window(windows)
        windows[1:] *= 100
        normalized_after = zscore_normalize_per_window(windows)
        np.testing.assert_allclose(normalized_before[0], normalized_after[0], atol=1e-6)

    def test_all_baselines_accept_stew_shape_and_backpropagate(self):
        x = torch.randn(2, 512, 14)
        y = torch.tensor([0, 1])
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                model = get_model(name, n_channels=14, n_timepoints=512, n_classes=2)
                logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 2))
                torch.nn.functional.cross_entropy(logits, y).backward()
                self.assertTrue(
                    all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                )

    def test_quality_report_is_zero_for_identical_data(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(8, 512, 14)).astype(np.float32)
        y = np.repeat([0, 1], 4)
        report = evaluate_synthetic_quality(x, y, x.copy(), y.copy())
        for metrics in report["classes"].values():
            self.assertEqual(metrics["channel_covariance_relative_error"], 0.0)
            self.assertEqual(metrics["lag1_autocorrelation_relative_error"], 0.0)
            self.assertTrue(
                all(error == 0.0 for error in metrics["band_power_relative_error"].values())
            )

    def test_subject_condition_metrics_weight_subjects_equally(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        groups = np.array([1, 1, 1, 1, 2, 2])
        probabilities = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9],
            [0.7, 0.3], [0.3, 0.7],
        ])
        metrics = evaluate_predictions(y, probabilities, groups, seed=42)
        self.assertEqual(metrics["window_level"]["accuracy"], 1.0)
        self.assertEqual(metrics["subject_condition_level"]["accuracy"], 1.0)
        self.assertEqual(
            metrics["subject_condition_level"]["n_subject_condition_units"], 4
        )

    def test_paired_sign_flip_test_detects_direction(self):
        result = paired_sign_flip_test([0.5, 0.6, 0.7], [0.6, 0.7, 0.8])
        self.assertAlmostEqual(result["mean_paired_delta"], 0.1)
        self.assertEqual(result["n_pairs"], 3)

    def test_reference_wgan_gp_adam_betas(self):
        self.assertEqual(ADAM_BETAS, (0.0, 0.9))

    def test_legacy_model_alias_resolves_to_adapted_model(self):
        model = get_model("eegnet", n_channels=14, n_timepoints=512)
        self.assertEqual(model.__class__.__name__, "EEGNetAdapted")

    def test_planned_datasets_are_not_accepted_by_training_cli_config(self):
        self.assertIn("iub", config.PLANNED_DATASETS)
        self.assertNotIn("iub", config.SUPPORTED_DATASETS)
        with self.assertRaises(ValueError):
            config.normalize_dataset_name("iub")

    def test_model_initialization_is_independent_of_execution_order(self):
        set_seed(42)
        first = get_model("1dcnn", n_channels=14, n_timepoints=512)
        first_state = {name: value.clone() for name, value in first.state_dict().items()}
        _ = get_model("lstm", n_channels=14, n_timepoints=512)
        set_seed(42)
        repeated = get_model("1dcnn", n_channels=14, n_timepoints=512)
        for name, value in repeated.state_dict().items():
            self.assertTrue(torch.equal(first_state[name], value))


if __name__ == "__main__":
    unittest.main()
