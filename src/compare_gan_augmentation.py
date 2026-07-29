"""
The research-grade with-GAN vs without-GAN comparison, for the paper.

Trains a SEPARATE CWGAN-GP inside EACH fold's training split only (never
sees that fold's held-out validation subjects -- avoids leakage), generates
synthetic data to augment that fold's training set, then trains the given
classifier on real-only vs real+synthetic. Uses the IDENTICAL GroupKFold
folds as train_baseline.py (same seed/order) so results are directly
comparable to your Phase 2 baseline numbers.

Within each fold, an inner validation split (carved from that fold's
training subjects only) is used for checkpoint selection -- the fold's real
held-out subjects are touched exactly once, for the final reported metric,
never during training or checkpoint selection. Synthetic data is added only
to the inner-training partition, never to inner-validation or the held-out
fold.

Usage:
    python3 -m src.compare_gan_augmentation --dataset stew --model 1dcnn --gan_epochs 200 --clf_epochs 30
"""

import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from . import config
from .augmentation import augmentation_counts_by_class, generate_simple_augmentation
from .evaluation import (
    holm_adjust_p_values,
    paired_sign_flip_test,
    paired_subject_cluster_test,
)
from .labeling import build_dataset
from .models import model_metadata
from .preprocessing import load_processed
from .provenance import build_provenance, write_json
from .reporting import (
    comparison_table_rows,
    upsert_csv,
    write_comparison_plot,
    write_comparison_diagnostic_plots,
    write_csv,
    write_learning_curve_plot,
)
from .reproducibility import set_seed
from .split import inner_group_split_indices
from .synthetic_quality import evaluate_synthetic_quality, save_synthetic_quality_plots
from .train_baseline import train_one_fold
from .train_gan import (
    ADAM_BETAS,
    GAN_LEARNING_RATE,
    LAMBDA_GP,
    N_CRITIC,
    QUALITY_SAMPLES_PER_CLASS,
    generate_synthetic,
    plot_loss_curve,
    train_gan,
)


def _load_fold_gan_cache(cache_path, expected_metadata: dict):
    with np.load(cache_path, allow_pickle=False) as cached:
        required = {
            "X_synth", "y_synth", "X_quality", "y_quality", "metadata_json"
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(
                f"GAN cache {cache_path} is incomplete (missing {sorted(missing)})."
            )
        metadata = json.loads(str(cached["metadata_json"].item()))
        mismatches = {
            key: {"expected": expected, "found": metadata.get(key)}
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"GAN cache {cache_path} does not match this protocol: {mismatches}. "
                "Choose a new --gan_cache_name."
            )
        return (
            cached["X_synth"].astype(np.float32),
            cached["y_synth"].astype(np.int64),
            cached["X_quality"].astype(np.float32),
            cached["y_quality"].astype(np.int64),
            metadata["training_history"],
        )


def _save_fold_gan_cache(
    cache_path,
    X_synth,
    y_synth,
    X_quality,
    y_quality,
    metadata,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_synth=X_synth,
        y_synth=y_synth,
        X_quality=X_quality,
        y_quality=y_quality,
        metadata_json=np.array(json.dumps(metadata)),
    )


def run_comparison(model_name: str, gan_epochs: int, clf_epochs: int,
                    n_folds: int = config.N_FOLDS, batch_size: int = 64,
                    synth_fraction: float = 0.25, dataset: str = config.DEFAULT_DATASET,
                    seed: int = config.RANDOM_SEED, run_name: str | None = None,
                    quality_samples_per_class: int = QUALITY_SAMPLES_PER_CLASS,
                    include_simple_augmentation: bool = False,
                    gan_cache_name: str | None = None,
                    overwrite: bool = False):
    """
    synth_fraction: synthetic samples added for each class relative to that
    class's real inner-training count. For example, 0.25 adds 25% per class.
    """
    dataset = config.normalize_dataset_name(dataset)
    if not np.isfinite(synth_fraction) or synth_fraction < 0:
        raise ValueError("synth_fraction must be finite and non-negative.")
    if quality_samples_per_class <= 0:
        raise ValueError("quality_samples_per_class must be positive.")
    run_name = run_name or f"gan_comparison_{model_name}_seed{seed}_frac{synth_fraction:g}"
    existing_paths = [
        path for path in (
            config.run_dir(dataset, run_name),
            config.model_dir(dataset, run_name),
            config.output_dir(dataset, run_name),
        ) if path.exists()
    ]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Comparison artifacts already exist; choose another --run_name or pass "
            "--overwrite: " + ", ".join(str(path) for path in existing_paths)
        )
    set_seed(seed)
    device = config.get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset}")

    processed = load_processed(dataset=dataset)
    X, y, groups = build_dataset(processed)
    print(f"Full dataset: X={X.shape}, y={y.shape}, subjects={len(set(groups))}")

    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(X, y, groups))  # SAME split logic as train_baseline.py (same seed/order)
    split_path = config.processed_dir(dataset) / "split.json"
    provenance = build_provenance(dataset, split_path if split_path.exists() else None)

    results = {
        "without_gan": [],
        "with_gan": [],
        "with_simple_augmentation": [],
        "metadata": {
            "dataset": dataset,
            "random_seed": seed,
            "n_folds": n_folds,
            "gan_epochs": gan_epochs,
            "classifier_epochs": clf_epochs,
            "batch_size": batch_size,
            "synth_fraction_per_class": synth_fraction,
            "quality_samples_per_class": quality_samples_per_class,
            "run_name": run_name,
            "include_simple_augmentation": include_simple_augmentation,
            "gan_cache_name": gan_cache_name,
        },
    }

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        groups_train = groups[train_idx]
        val_subjects = sorted(set(groups[val_idx].tolist()))
        print(f"\n{'='*60}\nFold {fold_i+1}/{n_folds} (held-out subjects: {val_subjects})\n{'='*60}")

        # Carve inner validation from this fold's TRAINING subjects only.
        # X_val/y_val (this fold's real held-out subjects) is never used for
        # checkpoint selection -- only for the final reported metric, once,
        # for both conditions below.
        inner_train_idx, inner_val_idx = inner_group_split_indices(groups_train, seed=seed)
        if inner_val_idx is None:
            raise ValueError("Strict GAN comparison requires at least three training subjects.")
        X_it, y_it = X_train[inner_train_idx], y_train[inner_train_idx]
        X_iv, y_iv = X_train[inner_val_idx], y_train[inner_val_idx]
        inner_train_subjects = sorted(np.unique(groups_train[inner_train_idx]).astype(int).tolist())
        inner_val_subjects = sorted(np.unique(groups_train[inner_val_idx]).astype(int).tolist())
        fold_run_dir = config.run_dir(dataset, run_name) / f"fold_{fold_i + 1}"
        fold_model_dir = config.model_dir(dataset, run_name) / f"fold_{fold_i + 1}"
        fold_output_dir = config.output_dir(dataset, run_name) / f"fold_{fold_i + 1}"

        # --- WITHOUT GAN: baseline on real inner-training data only ---
        classifier_seed = seed + fold_i
        set_seed(classifier_seed)
        print("  [without GAN] training...")
        metrics_no_gan, model_no_gan = train_one_fold(
            model_name, X_it, y_it, X_iv, y_iv, X_val, y_val,
            groups[val_idx], device=device, epochs=clf_epochs, batch_size=batch_size,
            seed=classifier_seed,
        )
        print(f"    acc={metrics_no_gan['accuracy']:.3f} f1_macro={metrics_no_gan['f1_macro']:.3f}")
        results["without_gan"].append(metrics_no_gan)

        gan_seed = seed + 10_000 + fold_i
        n_by_class = augmentation_counts_by_class(y_it, synth_fraction)
        cache_metadata = {
            "dataset": dataset,
            "fold": fold_i + 1,
            "base_seed": seed,
            "gan_seed": gan_seed,
            "gan_epochs": gan_epochs,
            "gan_batch_size": batch_size,
            "synth_fraction_per_class": synth_fraction,
            "quality_samples_per_class": quality_samples_per_class,
            "synthetic_class_counts": {
                "0": n_by_class[0],
                "1": n_by_class[1],
            },
            "n_timepoints": int(X_it.shape[1]),
            "n_channels": int(X_it.shape[2]),
            "inner_train_subjects": inner_train_subjects,
            "inner_val_subjects": inner_val_subjects,
            "holdout_subjects": val_subjects,
            "processed_subjects_sha256": provenance["processed_subjects_sha256"],
        }
        cache_path = (
            config.processed_dir(dataset)
            / "gan_cv_cache"
            / gan_cache_name
            / f"fold_{fold_i + 1}.npz"
            if gan_cache_name
            else None
        )
        cache_reused = bool(cache_path and cache_path.exists())
        gen = crit = None
        if cache_reused:
            print(f"  [with GAN] reusing validated fold cache: {cache_path}")
            X_synth, y_synth, X_quality, y_quality, history = _load_fold_gan_cache(
                cache_path, cache_metadata
            )
        else:
            # The GAN sees inner-training subjects only. Inner-validation and
            # held-out subjects are encoded in the cache metadata but never used
            # for GAN optimization.
            print("  [with GAN] training CWGAN-GP on inner-training subjects only...")
            gen, crit, history = train_gan(
                X_it,
                y_it,
                device=device,
                epochs=gan_epochs,
                batch_size=batch_size,
                seed=gan_seed,
            )
            print(
                f"  Generating augmentation fraction={synth_fraction:.2f}: "
                f"class0={n_by_class[0]}, class1={n_by_class[1]}..."
            )
            X_synth, y_synth = generate_synthetic(
                gen, n_by_class, device,
                n_timepoints=X_it.shape[1], n_channels=X_it.shape[2],
            )
            X_quality, y_quality = generate_synthetic(
                gen,
                {0: quality_samples_per_class, 1: quality_samples_per_class},
                device,
                n_timepoints=X_it.shape[1],
                n_channels=X_it.shape[2],
            )
            if cache_path:
                cache_payload = {
                    **cache_metadata,
                    "training_history": history,
                }
                _save_fold_gan_cache(
                    cache_path,
                    X_synth,
                    y_synth,
                    X_quality,
                    y_quality,
                    cache_payload,
                )
                cache_model_dir = (
                    config.model_dir(dataset, gan_cache_name)
                    / f"fold_{fold_i + 1}"
                )
                cache_model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    gen.state_dict(),
                    cache_model_dir / "cwgan_gp_generator.pt",
                )
                torch.save(
                    crit.state_dict(),
                    cache_model_dir / "cwgan_gp_critic.pt",
                )
                write_json(
                    config.run_dir(dataset, gan_cache_name)
                    / f"fold_{fold_i + 1}"
                    / "manifest.json",
                    {
                        "run_type": "fold_isolated_gan_cache",
                        **cache_payload,
                        "cache_path": str(cache_path.relative_to(config.PROJECT_ROOT)),
                        "model_directory": str(
                            cache_model_dir.relative_to(config.PROJECT_ROOT)
                        ),
                        "provenance": provenance,
                    },
                )
        gan_loss_path = (
            config.output_dir(dataset, gan_cache_name)
            / f"fold_{fold_i + 1}"
            / "gan_training_loss.png"
            if gan_cache_name
            else fold_output_dir / "gan_training_loss.png"
        )
        if not gan_loss_path.exists():
            gan_loss_path.parent.mkdir(parents=True, exist_ok=True)
            plot_loss_curve(history, gan_loss_path)
        quality = evaluate_synthetic_quality(
            X_it,
            y_it,
            X_quality,
            y_quality,
            fs=config.sampling_rate_hz(dataset),
            X_reference=X_iv,
            y_reference=y_iv,
        )
        save_synthetic_quality_plots(
            X_it,
            y_it,
            X_quality,
            y_quality,
            fs=config.sampling_rate_hz(dataset),
            output_dir=fold_output_dir,
            class_names=config.class_names(dataset),
        )

        if include_simple_augmentation:
            X_simple, y_simple = generate_simple_augmentation(
                X_it, y_it, n_by_class, seed=classifier_seed + 20_000
            )
            set_seed(classifier_seed)
            print("  [simple augmentation] training classifier...")
            metrics_simple, model_simple = train_one_fold(
                model_name,
                np.concatenate([X_it, X_simple]),
                np.concatenate([y_it, y_simple]),
                X_iv,
                y_iv,
                X_val,
                y_val,
                groups[val_idx],
                device=device,
                epochs=clf_epochs,
                batch_size=batch_size,
                seed=classifier_seed,
            )
            results["with_simple_augmentation"].append(metrics_simple)

        # IMPORTANT: synthetic data augments ONLY the inner-training
        # partition (X_it/y_it), never inner-val (X_iv/y_iv) and never the
        # real held-out fold (X_val/y_val). Checkpoint selection must be
        # judged on real data only.
        X_it_aug = np.concatenate([X_it, X_synth], axis=0)
        y_it_aug = np.concatenate([y_it, y_synth], axis=0)
        print(f"  Augmented inner-train set: {X_it.shape[0]} real + {X_synth.shape[0]} synthetic "
              f"= {X_it_aug.shape[0]} total")

        set_seed(classifier_seed)
        print("  [with GAN] training classifier on real+synthetic...")
        metrics_gan, model_gan = train_one_fold(
            model_name, X_it_aug, y_it_aug, X_iv, y_iv, X_val, y_val,
            groups[val_idx], device=device, epochs=clf_epochs, batch_size=batch_size,
            seed=classifier_seed,
        )
        print(f"    acc={metrics_gan['accuracy']:.3f} f1_macro={metrics_gan['f1_macro']:.3f}")
        results["with_gan"].append(metrics_gan)

        learning_curve_conditions = {
            "without_gan": metrics_no_gan,
            "with_gan": metrics_gan,
        }
        if include_simple_augmentation:
            learning_curve_conditions["with_simple_augmentation"] = metrics_simple
        write_learning_curve_plot(
            fold_output_dir / "classifier_learning_curves.png",
            learning_curve_conditions,
        )

        fold_model_dir.mkdir(parents=True, exist_ok=True)
        if gen is not None and not gan_cache_name:
            torch.save(gen.state_dict(), fold_model_dir / "cwgan_gp_generator.pt")
            torch.save(crit.state_dict(), fold_model_dir / "cwgan_gp_critic.pt")
        torch.save(model_no_gan.state_dict(), fold_model_dir / f"{model_name}_without_gan.pt")
        torch.save(model_gan.state_dict(), fold_model_dir / f"{model_name}_with_gan.pt")
        if include_simple_augmentation:
            torch.save(
                model_simple.state_dict(),
                fold_model_dir / f"{model_name}_with_simple_augmentation.pt",
            )
        write_json(fold_run_dir / "gan_training_history.json", history)
        write_json(fold_output_dir / "synthetic_quality.json", quality)
        write_json(fold_run_dir / "manifest.json", {
            "run_type": "paired_per_fold_gan_comparison",
            "dataset": dataset,
            "model": model_name,
            "model_metadata": model_metadata(model_name),
            "fold": fold_i + 1,
            "classifier_seed": classifier_seed,
            "gan_seed": gan_seed,
            "gan_epochs": gan_epochs,
            "gan_training_seconds": history["training_seconds"],
            "gan_mean_epoch_seconds": history["mean_epoch_seconds"],
            "classifier_epochs": clf_epochs,
            "batch_size": batch_size,
            "synth_fraction_per_class": synth_fraction,
            "synthetic_class_counts": {"0": n_by_class[0], "1": n_by_class[1]},
            "quality_samples_per_class": quality_samples_per_class,
            "gan_cache_name": gan_cache_name,
            "gan_cache_reused": cache_reused,
            "gan_cache_path": (
                str(cache_path.relative_to(config.PROJECT_ROOT))
                if cache_path
                else None
            ),
            "gan_training_loss_plot": str(
                gan_loss_path.relative_to(config.PROJECT_ROOT)
            ),
            "gan_optimizer": {
                "name": "Adam",
                "learning_rate": GAN_LEARNING_RATE,
                "betas": list(ADAM_BETAS),
                "n_critic": N_CRITIC,
                "lambda_gp": LAMBDA_GP,
            },
            "inner_train_subjects": inner_train_subjects,
            "inner_val_subjects": inner_val_subjects,
            "holdout_subjects": val_subjects,
            "without_gan_metrics": metrics_no_gan,
            "with_gan_metrics": metrics_gan,
            "with_simple_augmentation_metrics": (
                metrics_simple if include_simple_augmentation else None
            ),
            "synthetic_quality": quality,
            "quality_report": str((fold_output_dir / "synthetic_quality.json").relative_to(config.PROJECT_ROOT)),
            "model_directory": str(fold_model_dir.relative_to(config.PROJECT_ROOT)),
            "provenance": provenance,
        })

    results["paired_tests"] = {}
    results["subject_cluster_tests"] = {}
    for condition in ("with_gan", "with_simple_augmentation"):
        if not results[condition]:
            continue
        tests = {}
        for level, metric in (
            ("window", "accuracy"),
            ("window", "f1_macro"),
            ("subject_condition", "accuracy"),
            ("subject_condition", "f1_macro"),
        ):
            def values(source):
                if level == "window":
                    return [fold[metric] for fold in source]
                return [fold["subject_condition_level"][metric] for fold in source]

            tests[f"{level}_{metric}"] = paired_sign_flip_test(
                values(results["without_gan"]), values(results[condition])
            )
        adjusted = holm_adjust_p_values({
            name: test["exact_two_sided_p_value"] for name, test in tests.items()
        })
        for name, adjusted_p in adjusted.items():
            tests[name]["holm_adjusted_p_value"] = adjusted_p
        results["paired_tests"][condition] = tests
        cluster_tests = {
            metric: paired_subject_cluster_test(
                results["without_gan"],
                results[condition],
                metric=metric,
                seed=seed + 30_000,
            )
            for metric in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")
        }
        cluster_adjusted = holm_adjust_p_values({
            name: test["subject_cluster_randomization_p_value"]
            for name, test in cluster_tests.items()
        })
        for name, adjusted_p in cluster_adjusted.items():
            cluster_tests[name]["holm_adjusted_p_value"] = adjusted_p
        results["subject_cluster_tests"][condition] = cluster_tests
    write_json(config.run_dir(dataset, run_name) / "comparison_summary.json", results)
    table_rows = comparison_table_rows(results, model_name)
    write_csv(
        config.output_dir(dataset, run_name) / "comparison_table.csv",
        table_rows,
    )
    upsert_csv(
        config.output_dir(dataset) / "gan_comparison_master_table.csv",
        table_rows,
        key_fields=("run_name", "model", "augmentation", "level", "metric"),
    )
    write_comparison_plot(
        config.output_dir(dataset, run_name) / "comparison_performance.png",
        results,
    )
    write_comparison_diagnostic_plots(
        config.output_dir(dataset, run_name)
        / "comparison_classification_diagnostics.png",
        results,
    )
    return results


def summarize_comparison(results: dict, model_name: str):
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY: {model_name}")
    print(f"{'='*70}")
    print("Paired exact sign-flip tests:")
    for condition, tests in results["paired_tests"].items():
        print(f"  {condition} vs real-only:")
        for metric, test in tests.items():
            print(
                f"    {metric}: delta={test['mean_paired_delta']:+.4f}, "
                f"p={test['exact_two_sided_p_value']:.4f}, "
                f"Holm p={test['holm_adjusted_p_value']:.4f}"
            )
    print("Primary pooled subject-cluster tests:")
    for condition, tests in results["subject_cluster_tests"].items():
        for metric, test in tests.items():
            low, high = test["subject_cluster_bootstrap_95ci"]
            print(
                f"  {condition} {metric}: delta={test['paired_delta']:+.4f}, "
                f"95% CI=[{low:+.4f}, {high:+.4f}], "
                f"Holm p={test['holm_adjusted_p_value']:.4f}"
            )
    for condition in ("without_gan", "with_gan", "with_simple_augmentation"):
        if not results.get(condition):
            continue
        accs = [m["accuracy"] for m in results[condition]]
        f1s = [m["f1_macro"] for m in results[condition]]
        subject_accs = [m["subject_condition_level"]["accuracy"] for m in results[condition]]
        print(f"  {condition:14s}  accuracy = {np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
              f"f1_macro = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}   "
              f"subject-cond acc = {np.mean(subject_accs):.3f} +/- {np.std(subject_accs):.3f}")
    print(f"{'='*70}")


def save_comparison(results: dict, model_name: str, dataset: str = config.DEFAULT_DATASET,
                    seed: int = config.RANDOM_SEED):
    run_name = results.get("metadata", {}).get("run_name")
    filename = f"{run_name}.json" if run_name else f"gan_comparison_{model_name}_seed{seed}.json"
    out_path = config.output_dir(dataset) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    parser.add_argument("--model", type=str, default="1dcnn")
    parser.add_argument("--gan_epochs", type=int, default=200)
    parser.add_argument("--clf_epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--synth_fraction",
        type=float,
        default=0.25,
        help=(
            "Extra synthetic windows per class as a fraction of that class's "
            "real inner-training windows (default: 0.25)."
        ),
    )
    parser.add_argument("--quality_samples_per_class", type=int, default=QUALITY_SAMPLES_PER_CLASS)
    parser.add_argument("--include_simple_augmentation", action="store_true")
    parser.add_argument(
        "--gan_cache_name",
        type=str,
        default=None,
        help=(
            "Shared, protocol-validated per-fold GAN cache. Use the same name "
            "across classifier runs to train each fold GAN only once."
        ),
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = run_comparison(
        args.model, gan_epochs=args.gan_epochs, clf_epochs=args.clf_epochs,
        n_folds=args.folds, batch_size=args.batch_size, dataset=args.dataset,
        seed=args.seed, synth_fraction=args.synth_fraction, run_name=args.run_name,
        quality_samples_per_class=args.quality_samples_per_class,
        include_simple_augmentation=args.include_simple_augmentation,
        gan_cache_name=args.gan_cache_name,
        overwrite=args.overwrite,
    )
    summarize_comparison(results, args.model)
    save_comparison(results, args.model, dataset=args.dataset, seed=args.seed)
