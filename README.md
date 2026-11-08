# DINOv3 Under Extreme Domain Shift

**Deep Learning Course Project — MVA Master's**

**Author: Gloire LINVANI**

This project studies how **DINOv3** (Meta AI's latest self-supervised vision foundation model) generalizes across datasets exhibiting strong **domain shift** relative to its natural image pretraining data. We evaluate its features and fine-tuned performance on three highly domain-shifted benchmarks: histopathology images, satellite imagery, and underwater coral reef scenes.

---

## Overview

DINOv3 is pretrained on a large curated web dataset (LVD-1689M) using self-supervised learning. We probe its generalization by:

1. **Linear probing**: training only task-specific heads on top of frozen DINOv3 features.
2. **LoRA fine-tuning**: applying Low-Rank Adaptation (r=16, α=32) to the attention and MLP layers of the frozen backbone.
3. **Feature analysis**: comparing DINOv3 and DINOv2 last layer features via PCA projections and cosine similarity maps across all three domains.

---

## Datasets

| Dataset       | Task | Domain                              | Classes |
|---------------|---|-------------------------------------|---------|
| PatchCamelyon | Binary classification | Histopathology (H&E patches, 96×96) | 2       | 
| DeepGlobe     | Semantic segmentation | Satellite imagery (2448×2448)       | 7       |
| Coralscapes   | Semantic segmentation | Underwater coral reefs (1024×2048)  | 40      |

---

## Model Architecture

- **Backbone**: DINOv3 ViT-L/16, 300M parameters.
- **Classification head**: linear layer on the concatenation of the CLS token and mean-pooled patch tokens (2×1024 → 1).
- **Segmentation head**: multi-scale linear head using features from 4 evenly-spaced intermediate layers (layers 5, 11, 17, 23), fused via concatenation + BatchNorm + 1×1 conv + bilinear upsampling.
- **LoRA**: applied to `qkv`, `proj`, `fc1`, `fc2` linear layers of the transformer blocks.

---

## Project Structure

```
dinov3-transferability/
├── report.pdf                        # Project report
└── dinov3_domain_shift/
    ├── run_main_google_colab.ipynb   # Colab notebook: runs all experiments through main.py
    ├── main.py                       # CLI experiment runner
    ├── models.py                     # Backbone loading, LoRA, classification & segmentation heads
    ├── training.py                   # Trainer, losses (CE, Jaccard, Dice), evaluation metrics
    ├── domain_shift_datasets.py      # Dataset classes and download utilities
    ├── augmentations.py              # Domain-specific augmentations
    ├── visualization.py              # PCA and cosine similarity feature visualizations
    ├── requirements.txt
    ├── __init__.py
    ├── dinov3/                       # DINOv3 repository from Meta AI, as a git submodule
    ├── data/                         # Not included: downloaded datasets (PatchCamelyon, DeepGlobe, Coralscapes)
    ├── models/                       # Not included: DINOv3 ViT-L/16 weights (downloaded from Meta AI)
    └── outputs/                      # Not included: training outputs (checkpoints, logs) and figures of the qualitative analysis
```

---

## Running on Google Colab

To run all experiments on Google Colab, use the notebook below. It calls `main.py` the same way as in [Running via CLI](#running-via-cli), with the arguments used for each experiment:

```
run_main_google_colab.ipynb
```

This notebook handles:
- Mounting Google Drive and setting paths
- Installing dependencies
- Running training experiments (DINOv3 with and without LoRA) on PatchCamelyon, DeepGlobe, and Coralscapes
- Evaluating trained models on the test sets
- Generating feature analysis figures (PCA, cosine similarity, segmentation maps)

### Setup steps in the notebook

1. **DINOv3 weights**: Request access at [ai.meta.com/dinov3](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). Once granted, download the `dinov3_vitl16` weights and upload them to your Google Drive. Set the `weights` path in the notebook accordingly.

2. **DINOv3 repository**: It is included as a git submodule in `dinov3_domain_shift/dinov3/`, at commit `54694f7` of Meta AI's repository. A plain `git clone` or a ZIP download leaves that folder empty, so clone this project with `git clone --recurse-submodules`, place it in your Drive, then point `repo_dir` to that folder.

---

## Running via CLI

You can also run experiments directly using `main.py`:

```bash
# Download all datasets
python main.py --action download --data_root ./data

# Train DINOv3 (frozen backbone) on PatchCamelyon
python main.py --action train --dataset patchcamelyon \
    --weights /path/to/dinov3_vitl16.pth \
    --repo_dir ./dinov3 \
    --freeze_backbone

# Train DINOv3 with LoRA on DeepGlobe
python main.py --action train --dataset deepglobe --use_lora \
    --weights /path/to/dinov3_vitl16.pth \
    --repo_dir ./dinov3

# Run all experiments (download datasets + train + visualizations)
python main.py --action all \
    --weights /path/to/dinov3_vitl16.pth \
    --repo_dir ./dinov3

# Generate feature visualizations only
python main.py --action visualize \
    --weights /path/to/dinov3_vitl16.pth \
    --repo_dir ./dinov3
```

Key CLI arguments:

| Argument | Description | Default |
|---|---|---|
| `--action` | `download`, `train`, `visualize`, or `all` | `all` |
| `--dataset` | `patchcamelyon`, `deepglobe`, `coralscapes`, or `all` | `all` |
| `--use_lora` | Enable LoRA fine-tuning | `False` |
| `--freeze_backbone` | Freeze backbone parameters | `False` |
| `--weights` | Path to DINOv3 checkpoint | — |
| `--repo_dir` | Path to local DINOv3 repository | — |
| `--data_root` | Path to data directory | `./data` |
| `--output_dir` | Path to output directory | `./outputs` |
| `--resume_training` | Resume from last checkpoint | `False` |
| `--checkpoint_path` | Path to checkpoint for resuming | — |
| `--early_stopping_patience` | Epochs without improvement before stopping | `10` |
| `--seed` | Random seed | `42` |

---

## Outputs

Training produces per-experiment directories under `--output_dir`:

```
outputs/
└── {dataset}_{model}[_lora]_{timestamp}/
    ├── checkpoint_best.pt       # Best model checkpoint (by validation metric)
    ├── checkpoint_latest.pt       # Last epoch checkpoint
    ├── history.json    # Per-epoch train/val losses and val metrics 
    └── experiment_summary.json  # Training configuration and best model test metrics 
```

Figures produced by the qualitative analysis experiments are saved under `outputs/visualizations/`:

```
visualizations/
├── patchcamelyon 
├── deepglobe
├── coralscapes
├── deepglobe_segmentation
└── coralscapes_segmentation 
```
---

## References

- **DINOv3**: Oriane Siméoni et al. DINOv3. 2025. arXiv: 2508.10104 [cs.CV]. URL: https://arxiv.org/abs/2508.10104.
- **DINOv2**: Maxime Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. 2024. arXiv: 2304.07193 [cs.CV]. URL: https://arxiv.org/abs/2304.07193.
- **PatchCamelyon**: Bastiaan S. Veeling et al. Rotation Equivariant CNNs for Digital Pathology. 2018. arXiv: 1806.03962 [cs.CV]. URL: https://arxiv.org/abs/1806.03962.
- **DeepGlobe**: Ilke Demir et al. “DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images”. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. June 2018.
- **Coralscapes**: Jonathan Sauder et al. The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs. 2025. arXiv: 2503.20000 [cs.CV]. URL: https://arxiv.org/abs/2503.20000.