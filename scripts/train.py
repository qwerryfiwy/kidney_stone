"""Single-run training script (Section 3.2, Colour 2D Mixed Data protocol).

Initializes training with a stratified 80/10/10 split, applies early
stopping on validation AUC, and performs a final evaluation on the held-out test split.
"""
from __future__ import annotations

import argparse
import json
import os
import torch
import yaml

from data.dataset import build_stratified_split_loaders
from models.enfm import build_model
from training.engine import fit, evaluate, TrainConfig
from training.seed import set_seed
from utils.checkpoint import load_checkpoint
from utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train ENFM or a baseline model on a single stratified split."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config).",
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
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.lr is not None:
        config["optim"]["lr"] = args.lr
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.data_root:
        config["data"]["root"] = args.data_root

    # Set seed
    seed = config["train"]["seed"]
    set_seed(seed)

    # Setup logging
    out_dir = config["logging"]["out_dir"]
    model_name = config["model"]["name"]
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(name="enfm", out_dir=out_dir, filename=f"train_{model_name}_seed{seed}.log")

    logger.info("=" * 60)
    logger.info(f"Starting single run training protocol for model: {model_name}")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(
        name=model_name,
        pyramid_channels=config["model"]["pyramid_channels"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_fc1=config["model"]["dropout_fc1"],
        dropout_fc2=config["model"]["dropout_fc2"],
    )
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Load loaders
    logger.info(f"Building data loaders from: {config['data']['root']}")
    train_loader, val_loader, test_loader = build_stratified_split_loaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        num_workers=config["data"]["num_workers"],
        seed=seed,
    )

    # Compile TrainConfig
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

    checkpoint_path = os.path.join(out_dir, f"best_model_{model_name}_seed{seed}.pt")
    logger.info(f"Training history will be recorded and checkpoints saved to {checkpoint_path}")

    # Train model
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=train_cfg,
        checkpoint_path=checkpoint_path,
    )

    # Load best model for evaluation
    logger.info(f"Loading best model checkpoint for evaluation...")
    load_checkpoint(checkpoint_path, model, device=device)

    # Evaluate on validation split
    val_results = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), device)
    logger.info(f"Best Epoch {history.best_epoch} Validation Metrics:")
    for metric, val in val_results.items():
        if isinstance(val, float):
            logger.info(f"  {metric}: {val:.4f}")
        else:
            logger.info(f"  {metric}: {val}")

    # Evaluate on held-out test split
    test_results = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    logger.info(f"Best Epoch {history.best_epoch} Test Metrics:")
    for metric, val in test_results.items():
        if isinstance(val, float):
            logger.info(f"  {metric}: {val:.4f}")
        else:
            logger.info(f"  {metric}: {val}")

    # Save training history and evaluation results to a JSON file
    run_info = {
        "config": config,
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_metrics": history.val_metrics,
            "best_epoch": history.best_epoch,
            "best_val_auc": history.best_val_auc,
        },
        "val_results": val_results,
        "test_results": test_results,
    }
    history_path = os.path.join(out_dir, f"results_{model_name}_seed{seed}.json")
    with open(history_path, "w") as f:
        json.dump(run_info, f, indent=4)
    logger.info(f"Results saved to {history_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
