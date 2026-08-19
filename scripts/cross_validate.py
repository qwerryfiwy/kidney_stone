"""Repeated k-fold cross-validation script (Section 3.2, repeated 5-fold CV protocol).

Executes training and evaluation across multiple folds and seeds, reporting
consolidated metric averages (accuracy, sensitivity, specificity, F1, NPV, AUC).
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
import yaml

from data.dataset import build_kfold_splits, build_fold_loaders
from models.enfm import build_model
from training.engine import fit, evaluate, TrainConfig
from training.seed import set_seed
from utils.checkpoint import load_checkpoint
from utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated stratified K-fold cross-validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
        help="List of random seeds to evaluate over (default: [42, 123, 999]).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs per fold training (overrides config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Data root directory (overrides config).",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with arguments
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.lr is not None:
        config["optim"]["lr"] = args.lr
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.data_root:
        config["data"]["root"] = args.data_root

    model_name = config["model"]["name"]
    out_dir = config["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Setup cross-validation logger
    logger = setup_logger(
        name="enfm_cv",
        out_dir=out_dir,
        filename=f"cv_{model_name}.log",
    )

    logger.info("=" * 60)
    logger.info(f"Starting repeated CV protocol for model: {model_name}")
    logger.info(f"Folds: {args.folds} | Seeds: {args.seeds}")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cv_runs = []

    for seed in args.seeds:
        logger.info(f"\n--- Running Seed: {seed} ---")
        set_seed(seed)

        # Generate K-fold index splits
        splits = build_kfold_splits(
            root=config["data"]["root"],
            n_folds=args.folds,
            seed=seed,
        )

        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Fold {fold+1}/{args.folds} (Seed {seed})")

            # Build fold loaders (which splits out a validation fraction internally)
            train_loader, val_loader, test_loader = build_fold_loaders(
                root=config["data"]["root"],
                train_idx=train_idx,
                test_idx=test_idx,
                image_size=config["data"]["image_size"],
                batch_size=config["data"]["batch_size"],
                num_workers=config["data"]["num_workers"],
                seed=seed,
            )

            # Build fresh model
            model = build_model(
                name=model_name,
                pyramid_channels=config["model"]["pyramid_channels"],
                num_classes=config["model"]["num_classes"],
                pretrained=config["model"]["pretrained"],
                dropout_fc1=config["model"]["dropout_fc1"],
                dropout_fc2=config["model"]["dropout_fc2"],
            )
            model.to(device)

            train_cfg = TrainConfig(
                epochs=config["train"]["epochs"],
                early_stopping_patience=config["train"]["early_stopping_patience"],
                amp=config["train"]["amp"],
                grad_clip_norm=config["train"]["grad_clip_norm"],
                lr=config["optim"]["lr"],
                weight_decay=config["optim"]["weight_decay"],
                scheduler_factor=config["optim"]["scheduler_factor"],
                scheduler_patience=config["optim"]["scheduler_patience"],
            )

            checkpoint_path = os.path.join(
                out_dir, f"cv_{model_name}_seed{seed}_fold{fold}.pt"
            )

            # Fit the model
            history = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                cfg=train_cfg,
                checkpoint_path=checkpoint_path,
            )

            # Evaluate on held-out test fold using the best model checkpoint
            load_checkpoint(checkpoint_path, model, device=device)
            test_results = evaluate(
                model, test_loader, torch.nn.CrossEntropyLoss(), device
            )

            logger.info(
                f"Fold {fold+1} Test Metrics: AUC={test_results['auc']:.4f} "
                f"Acc={test_results['accuracy']:.4f} F1={test_results['f1']:.4f}"
            )

            test_results["seed"] = seed
            test_results["fold"] = fold
            cv_runs.append(test_results)

    # Calculate aggregate metrics
    logger.info("\n" + "=" * 60)
    logger.info(f"CROSS-VALIDATION RESULTS SUMMARY FOR {model_name.upper()}")
    logger.info(f"Total evaluated runs: {len(cv_runs)}")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
    logger.info("-" * 60)

    summary_stats = {}
    metrics_to_report = [
        "accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "npv",
        "auc",
    ]
    for metric in metrics_to_report:
        vals = [run[metric] for run in cv_runs if not np.isnan(run[metric])]
        mean_val = float(np.mean(vals)) if vals else 0.0
        std_val = float(np.std(vals)) if vals else 0.0
        logger.info(f"{metric:<15} | {mean_val:.4f} | {std_val:.4f}")
        summary_stats[metric] = {"mean": mean_val, "std": std_val}
    logger.info("=" * 60)

    # Save CV results to a JSON file
    cv_info = {
        "config": config,
        "summary": summary_stats,
        "runs": cv_runs,
    }
    results_path = os.path.join(out_dir, f"cv_results_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(cv_info, f, indent=4)
    logger.info(f"Aggregated CV results saved to {results_path}")
    logger.info("Cross-validation protocol complete.")


if __name__ == "__main__":
    main()
