# ENFM: EfficientNet-FPN-Max for Kidney Stone Classification

Modular PyTorch implementation of **EfficientNet-FPN-Max (ENFM)** — an
EfficientNet-B0 backbone + top-down Feature Pyramid Network + attention-gated
bottom-up path aggregation + multi-level global max pooling — for binary
kidney-stone classification on non-contrast CT slices, as described in
*"Extremal Activation Preservation using Attention-Gated Feature Pyramids
for Kidney Stone Classification in Non-Contrast Computed Tomography."*

## Repo layout

```
enfm-kidney-stone/
├── configs/
│   └── default.yaml            # all hyperparameters in one place
├── data/
│   ├── dataset.py               # CTStoneDataset (ImageFolder-style loader)
│   └── transforms.py            # train/val/test augmentation pipelines
├── models/
│   ├── backbone.py              # EfficientNet-B0 feature extractor (C3,C4,C5)
│   ├── fpn.py                   # top-down Feature Pyramid Network
│   ├── attention_gate.py        # attention-gated bottom-up path aggregation
│   ├── pooling.py               # Max / Avg / Dual / multi-level pooling heads
│   ├── classifier.py            # fusion + classification head
│   ├── enfm.py                  # assembles backbone+FPN+gate+pool+head
│   └── baselines.py             # ResNet50 / MobileNetV3 / EfficientNet-V2S/B0 wrappers
├── training/
│   ├── engine.py                 # train_one_epoch / evaluate loops
│   ├── losses.py                 # cross-entropy wrapper
│   └── seed.py                   # reproducibility helpers
├── evaluation/
│   ├── metrics.py                 # accuracy, F1, AUC, sensitivity, specificity, NPV
│   ├── stats.py                   # Friedman test + pairwise paired t-tests
│   └── gradcam.py                 # Grad-CAM on the n5 pyramid level
├── utils/
│   ├── checkpoint.py
│   └── logging.py
├── scripts/
│   ├── train.py                   # single run: python scripts/train.py --config configs/default.yaml
│   ├── cross_validate.py          # 5-fold CV x N seeds protocol used in the paper
│   └── evaluate.py                # evaluate a checkpoint on a held-out split
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# single stratified-split run (Colour 2D Mixed Data protocol)
python scripts/train.py --config configs/default.yaml

# repeated 5-fold CV protocol (Grayscale / Colour Large protocol)
python scripts/cross_validate.py --config configs/default.yaml --folds 5 --seeds 42 123 999

# evaluate a saved checkpoint
python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/best_model.pt
```

Point `data.root` in `configs/default.yaml` to a directory with the standard
`ImageFolder` layout:

```
data_root/
├── stone/
│   └── *.png
└── non_stone/
    └── *.png
```

## Model variants

`models/enfm.py` exposes `build_model(name, ...)` where `name` is one of:
`enfm` (full attention-gated max model), `fpn_avg`, `fpn_gmp`, `fpn_dual`
(ablations), or a baseline key from `models/baselines.py`
(`resnet50`, `mobilenetv3`, `efficientnet_v2s`, `efficientnet_b0`).
This lets `scripts/train.py --model <name>` reproduce every row of the
paper's comparison tables with the same training loop.
