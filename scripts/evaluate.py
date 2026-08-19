"""Evaluation script (Section 3.2).

Loads a saved model checkpoint and evaluates it on a specified split of the
stratified 80/10/10 data partition (defaulting to the held-out test split).
"""
from __future__ import annotations

import argparse
import os
import torch
import yaml

from data.dataset import build_stratified_split_loaders
from models.enfm import build_model
from training.engine import evaluate
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model checkpoint on a split."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config).",
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
    if args.data_root:
        config["data"]["root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model_name = config["model"]["name"]
    model = build_model(
        name=model_name,
        pyramid_channels=config["model"]["pyramid_channels"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,  # We will load weights
        dropout_fc1=config["model"]["dropout_fc1"],
        dropout_fc2=config["model"]["dropout_fc2"],
    )
    model.to(device)

    # Load weights
    print(f"Loading checkpoint from: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)

    # Load loaders
    print(f"Building data loaders from: {config['data']['root']}")
    train_loader, val_loader, test_loader = build_stratified_split_loaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        num_workers=config["data"]["num_workers"],
        seed=config["train"]["seed"],
    )

    if args.split == "test":
        loader = test_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = train_loader

    print(f"Evaluating model on '{args.split}' split ({len(loader.dataset)} samples)...")
    results = evaluate(
        model=model,
        loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
    )

    print("=" * 60)
    print(f"EVALUATION RESULTS ({model_name.upper()} on {args.split.upper()} split)")
    print("=" * 60)
    for metric, val in results.items():
        if isinstance(val, float):
            print(f"  {metric:<15}: {val:.4f}")
        else:
            print(f"  {metric:<15}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
