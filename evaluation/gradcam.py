"""Grad-CAM implementation (Section 3.6):
Generates class activation maps on the n5 pyramid level for ENFM, or the
final conv layer for baseline models.
"""
from __future__ import annotations

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.enfm import build_model
from utils.checkpoint import load_checkpoint


class GradCAM:
    """Grad-CAM class that registers hooks to capture activations and gradients

    for a specified target layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        # Register forward and backward hooks
        self.handlers.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.handlers.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Computes the Grad-CAM heatmap for a target class.

        Args:
            x: Input tensor of shape (1, C, H, W).
            class_idx: Index of target class. If None, uses model's prediction.
        """
        self.model.eval()
        self.gradients = None
        self.activations = None

        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]

        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Failed to capture gradients or activations. Make sure the target layer is correct."
            )

        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(-2, -1), keepdim=True)  # (1, C, 1, 1)

        # Compute weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)

        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Interpolate to the input image size
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for h in self.handlers:
            h.remove()


def find_target_layer(model: nn.Module) -> nn.Module:
    """Attempts to find the target layer for Grad-CAM.

    For ENFM variants, target is the n5 level.
    For baselines, target is the last convolutional layer.
    """
    # 1. ENFM variants (gate5 is the n5 pyramid layer block)
    if hasattr(model, "bottom_up") and model.bottom_up is not None:
        if hasattr(model.bottom_up, "gate5"):
            return model.bottom_up.gate5

    # 2. ResNet50
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    # 3. MobileNetV3 / EfficientNet
    if hasattr(model, "conv_head"):
        return model.conv_head

    # 4. Fallback search for the last Conv2D layer in any submodules
    conv_layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append(m)
    if conv_layers:
        return conv_layers[-1]

    raise ValueError("Could not automatically determine target layer for Grad-CAM.")


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, cmap_name: str = "jet"
) -> np.ndarray:
    """Overlays the Grad-CAM heatmap on the input image using matplotlib colormaps."""
    if image.max() > 1.0:
        image = image / 255.0
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=-1)

    # Get colormap
    colormap = cm.get_cmap(cmap_name)
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Discard alpha channel

    # Linear blend
    overlaid = (1 - alpha) * image + alpha * heatmap_colored
    return (overlaid * 255).astype(np.uint8)


def save_visualization(
    save_path: str,
    original_img: np.ndarray,
    heatmap: np.ndarray,
    overlaid: np.ndarray,
) -> None:
    """Saves a side-by-side plot comparing original, heatmap, and overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlaid)
    axes[2].set_title("Overlaid Visualization")
    axes[2].axis("off")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM on a saved checkpoint and an input slice."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to input image."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/gradcam_result.png",
        help="Path to save the output visualization.",
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=1,
        help="Target class index for Grad-CAM (default: 1 for stone).",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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
    load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Load and preprocess image
    image_size = config["data"]["image_size"]
    img = Image.open(args.image).convert("RGB")
    img_resized = img.resize((image_size, image_size))
    original_np = np.array(img_resized)

    # Imports to build evaluation transforms
    from data.transforms import build_eval_transforms

    transform = build_eval_transforms(image_size)
    tensor = transform(img).unsqueeze(0).to(device)

    # Determine target layer and instantiate Grad-CAM
    target_layer = find_target_layer(model)
    print(f"Hooking target layer: {target_layer.__class__.__name__}")

    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    try:
        heatmap = gradcam(tensor, class_idx=args.class_idx)
        overlaid = overlay_heatmap(original_np, heatmap)

        # Save plot
        save_visualization(args.output, original_np, heatmap, overlaid)
        print(f"Grad-CAM visualization saved to: {args.output}")
    finally:
        gradcam.remove_hooks()


if __name__ == "__main__":
    main()
