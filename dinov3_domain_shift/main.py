"""
Main Experiment Runner
======================
Run experiments for DINOv3 domain shift project
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from augmentations import get_patchcamelyon_transforms, get_deepglobe_transforms, get_coralscapes_transforms
from domain_shift_datasets import (
    download_all_datasets,
    create_dataloaders,
    PatchCamelyonDataset,
    DeepGlobeDataset,
    CoralscapesDataset,
    CORALSCAPES_NUM_CLASSES,
)
from models import (
    create_model,
    load_dinov2,
    load_dinov3,
    count_parameters,
)
from training import (
    Trainer,
)
from visualization import (
    compare_models_pca,
    compare_models_cosine_similarity,
    compare_segmentation_maps,
)

# ==========================
# Experiment Configuration
# ==========================

EXPERIMENTS = {
    "patchcamelyon": {
        "task": "classification",
        "num_classes": 2,
        "batch_size": 64,
        "epochs": 30,
        "image_size": None,  # We use the initial 96x96 size
        "image_size_visualizations": 98,  # for visualization
        "learning_rate": 1e-3,
        "lora_learning_rate": 1e-4,
    },
    "deepglobe": {
        "task": "segmentation",
        "num_classes": 7,
        "batch_size": 8,
        "epochs": 50,
        "image_size": 512,  # or 896, training image size
        "image_size_visualizations": 512,  # for visualization
        "learning_rate": 1e-3,
        "lora_learning_rate": 1e-4,
    },
    "coralscapes": {
        "task": "segmentation",
        "num_classes": CORALSCAPES_NUM_CLASSES,
        "batch_size": 8,
        "epochs": 50,
        "image_size": (512, 1024),  # (256, 512) # height, width
        "image_size_visualizations": (512, 1024),  # for visualization
        "learning_rate": 1e-3,
        "lora_learning_rate": 1e-4,
    },
}


# =============================
# Main Experiment Functions
# =============================

def run_experiment(
        dataset_name: str,
        model_type: str = "dinov3",
        freeze_backbone: bool = True,
        use_lora: bool = False,
        ce_weight: float = 1 / 2,
        jaccard_weight: float = 1 / 2,
        dice_weight: float = 0.0,
        ignore_index: int = 255,
        data_root: Path = Path("./data"),
        output_dir: Path = Path("./outputs"),
        device: str = "cuda",
        seed: int = 42,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        resume_training: bool = False,
        checkpoint_path: Path | None = None,
        epochs_resume_training: int = 0,
        weights: str = '',
        repo_dir: str = ''
) -> Dict:
    """
    Run a single experiment
    
    Args:
        dataset_name: "patchcamelyon", "deepglobe", or "coralscapes"
        model_type: "dinov3"
        freeze_backbone: Whether to freeze the backbone during training
        use_lora: Whether to use LoRA fine-tuning,
        ce_weight: Weight for cross-entropy loss
        jaccard_weight: Weight for Jaccard loss
        dice_weight: Weight for Dice loss
        ignore_index: Index to ignore in segmentation loss
        data_root: Path to data directory
        output_dir: Path to output directory
        device: Device for computation
        seed: Random seed
        early_stopping_patience: Number of epochs without improvement before early stopping
        early_stopping_min_delta: Minimum change in validation metric to qualify as an improvement
        resume_training: Whether to resume training from last checkpoint if available
        checkpoint_path: Path to checkpoint to resume from
        epochs_resume_training: Target total number of epochs (including previous training) when resuming training
        weights: Path to frozen backbone weights for DINOv3
        repo_dir: Path to the repository directory for DINOv3 code
    
    Returns:
        Dictionary with experiment results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_config = EXPERIMENTS[dataset_name]

    exp_name = f"{dataset_name}_{model_type}"
    if use_lora:
        exp_name += "_lora"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {exp_name}")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"LoRA: {use_lora}")
    print(f"Output: {exp_dir}")
    print(f"{'=' * 60}\n")

    dataloaders = create_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=exp_config["batch_size"],
        image_size=exp_config["image_size"],
    )

    # LoRA configuration
    lora_config = None
    if use_lora:
        lora_config = {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["qkv", "proj", "fc1", "fc2"],
        }

    # Create model
    model = create_model(
        model_type=model_type,
        task=exp_config["task"],
        num_classes=exp_config["num_classes"],
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_config=lora_config,
        device=device,
        weights=weights,
        repo_dir=repo_dir
    )

    # Print model info
    param_counts = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {param_counts['total']}")
    print(f"  Trainable: {param_counts['trainable']}")
    print(f"  Frozen: {param_counts['frozen']}")
    print(f"  Trainable %: {param_counts['trainable_pct']:.2f}%\n")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        task=exp_config["task"],
        ce_weight=ce_weight,
        jaccard_weight=jaccard_weight,
        dice_weight=dice_weight,
        ignore_index=ignore_index,
        num_classes=exp_config["num_classes"],
        learning_rate=exp_config["learning_rate"],
        epochs=exp_config["epochs"],
        device=device,
        output_dir=exp_dir,
        use_lora=use_lora,
        lora_lr=exp_config["lora_learning_rate"] if use_lora else None,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    if resume_training and checkpoint_path is not None and checkpoint_path.exists():
        # Resume training from checkpoint
        # retrieve experiment directory from checkpoint path
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        exp_dir = checkpoint_path.parent
        trainer.output_dir = exp_dir
        history = trainer.resume_training(checkpoint_path=checkpoint_path, total_epochs=epochs_resume_training)
    else:
        # Train from scratch
        history = trainer.train()

    # Save experiment config
    exp_summary = {
        "dataset": dataset_name,
        "model_type": model_type,
        "use_lora": use_lora,
        "config": exp_config,
        "param_counts": param_counts,
        "best_epoch": trainer.best_epoch,
        "best_metric": trainer.best_metric,
        "test_metrics": history.get("test_metrics"),
    }

    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    print(f"\nExperiment complete.")
    print(f"Results saved to: {exp_dir}")

    return exp_summary


def run_all_experiments(
        data_root: Path = Path("./data"),
        output_dir: Path = Path("./outputs"),
        freeze_backbone: bool = True,
        ce_weight: float = 1 / 2,
        jaccard_weight: float = 1 / 2,
        dice_weight: float = 0.0,
        ignore_index: int = 255,
        device: str = "cuda",
        seed: int = 42,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        weights: str = '',
        repo_dir: str = ''
) -> Dict[str, Any]:
    """
    Run all experiments: We only train DINOv3, with and without LoRA
    
    """
    results = {}

    datasets = ["patchcamelyon", "deepglobe", "coralscapes"]

    for dataset in datasets:
        # DINOv3 without LoRA fine-tuning
        result = run_experiment(
            dataset_name=dataset,
            model_type="dinov3",
            use_lora=False,
            data_root=data_root,
            output_dir=output_dir,
            ce_weight=ce_weight,
            jaccard_weight=jaccard_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
            freeze_backbone=freeze_backbone,
            device=device,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            weights=weights,
            repo_dir=repo_dir
        )
        results[f"{dataset}_dinov3_frozen"] = result

        # DINOv3 with LoRA fine-tuning
        result = run_experiment(
            dataset_name=dataset,
            model_type="dinov3",
            use_lora=True,
            data_root=data_root,
            output_dir=output_dir,
            ce_weight=ce_weight,
            jaccard_weight=jaccard_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
            freeze_backbone=freeze_backbone,
            device=device,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            weights=weights,
            repo_dir=repo_dir
        )
        results[f"{dataset}_dinov3_lora"] = result

    # Save combined results
    results_path = Path(output_dir) / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("All experiments complete.")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 60}")

    return results


# =============================
# Visualization Functions
# =============================

def run_feature_visualization(
        data_root: Path = Path("./data"),
        output_dir: Path = Path("./visualizations"),
        device: str = "cuda",
        num_samples: int = 5,
        weights: str = '',
        repo_dir: str = '',
        seed: int = 42
) -> None:
    """
    Run feature visualization comparing DINOv2 and DINOv3
    Creates PCA and cosine similarity visualizations

    Args:
        data_root: Path to data directory
        output_dir: Path to output directory
        device: Device for computation
        num_samples: Number of samples to visualize
        weights: Path to frozen backbone weights
        repo_dir: Path to the repository directory for DINOv3 code
        seed: Random seed for reproducible sample selection
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Running Feature Visualization")
    print(f"{'=' * 60}\n")

    # Load models
    print("Loading DINOv2...")
    dinov2 = load_dinov2("dinov2_vitl14")
    dinov2 = dinov2.to(device)
    dinov2.eval()

    print("Loading DINOv3...")
    dinov3 = load_dinov3("dinov3_vitl16", weights=weights, repo_dir=repo_dir)
    dinov3 = dinov3.to(device)
    dinov3.eval()

    models = {"DINOv2": dinov2, "DINOv3": dinov3}
    num_layers = {"DINOv2": 24, "DINOv3": 24}
    patch_sizes = {"DINOv2": 14, "DINOv3": 16}

    # Get sample images from each dataset
    datasets_to_visualize = {
        "patchcamelyon": {
            "kwargs": {"split": "test", "image_size": EXPERIMENTS["patchcamelyon"]["image_size_visualizations"],
                       "transform": get_patchcamelyon_transforms(is_train=False)},
            "DatasetClass": PatchCamelyonDataset,
        },
        "deepglobe": {
            "kwargs": {"split": "test", "image_size": EXPERIMENTS["deepglobe"]["image_size_visualizations"],
                       "transform": get_deepglobe_transforms(is_train=False)},
            "DatasetClass": DeepGlobeDataset,
        },
        "coralscapes": {
            "kwargs": {"split": "test", "image_size": EXPERIMENTS["coralscapes"]["image_size_visualizations"],
                       "transform": get_coralscapes_transforms(is_train=False)},
            "DatasetClass": CoralscapesDataset,
        },
    }

    for dataset_name, dataset_config in datasets_to_visualize.items():
        print(f"\nVisualizing backbone features for {dataset_name}...")

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        try:
            DatasetClass = dataset_config["DatasetClass"]
            dataset = DatasetClass(
                data_root=data_root,
                **dataset_config["kwargs"],
            )

            # Randomly select indices
            np.random.seed(seed)
            num_to_sample = min(num_samples, len(dataset))
            random_indices = np.random.choice(len(dataset), size=num_to_sample, replace=False)

            # Collect all images first
            images = []
            for dataset_idx in random_indices:
                image, _ = dataset[int(dataset_idx)]

                # Convert tensor to PIL if needed
                if isinstance(image, torch.Tensor):
                    from visualization import denormalize_image
                    image = denormalize_image(image)
                    image = Image.fromarray(
                        (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    )

                images.append(image)

            # For Coralscapes, we save each image separately due to aspect ratio
            if dataset_name == "coralscapes":
                for idx, image in enumerate(images):
                    # PCA comparison - single image
                    fig = compare_models_pca(
                        models=models,
                        images=image,
                        num_layers=num_layers,
                        patch_sizes=patch_sizes,
                        device=device,
                        save_path=dataset_dir / f"pca_comparison_{idx}.pdf",
                        image_size=dataset_config['kwargs']['image_size'],
                        dataset_name=dataset_name,
                        idx=idx
                    )
                    plt.close(fig)

                    # Cosine similarity comparison - single image
                    fig = compare_models_cosine_similarity(
                        models=models,
                        images=image,
                        query_points=None,
                        num_layers=num_layers,
                        patch_sizes=patch_sizes,
                        device=device,
                        save_path=dataset_dir / f"cosine_similarity_{idx}.pdf",
                        image_size=dataset_config['kwargs']['image_size'],
                        dataset_name=dataset_name,
                        idx=idx
                    )
                    plt.close(fig)
            else:
                # For other datasets, we combine all images in one PDF
                # PCA comparison
                fig = compare_models_pca(
                    models=models,
                    images=images,
                    num_layers=num_layers,
                    patch_sizes=patch_sizes,
                    device=device,
                    save_path=dataset_dir / "pca_comparison.pdf",
                    image_size=dataset_config['kwargs']['image_size'],
                    dataset_name=dataset_name,
                )
                plt.close(fig)

                # Cosine similarity comparison
                fig = compare_models_cosine_similarity(
                    models=models,
                    images=images,
                    query_points=None,  # Will default to center for each image
                    num_layers=num_layers,
                    patch_sizes=patch_sizes,
                    device=device,
                    save_path=dataset_dir / "cosine_similarity.pdf",
                    image_size=dataset_config['kwargs']['image_size'],
                    dataset_name=dataset_name,
                )
                plt.close(fig)

        except Exception:
            print(f"Error visualizing {dataset_name}")
            raise

    print(f"\nVisualizations saved to: {output_dir}")


def run_segmentation_visualization(
        dataset_name: str,
        data_root: Path = Path("./data"),
        output_dir: Path = Path("./visualizations"),
        device: str = "cuda",
        num_samples: int = 4,
        image_size: int | Tuple[int, int] = 512,
        overlay: bool = False,
        overlay_alpha: float = 0.5,
        model_no_lora_path: str = '',
        model_lora_path: str = '',
        repo_dir: str = '',
        weights: str = '',
        seed: int = 42
) -> None:
    """
    Run segmentation comparison visualization for trained models.

    Creates:
    - Segmentation map comparison (raw predictions)
    - Overlay comparison (predictions overlaid on images)

    Args:
        dataset_name: Name of the dataset ("deepglobe" or "coralscapes")
        data_root: Path to data directory
        output_dir: Path to output directory
        device: Device for computation
        num_samples: Number of samples to visualize
        image_size: Image size used during training
        overlay: Whether to create overlay visualizations
        overlay_alpha: Alpha value for overlay transparency
        model_no_lora_path: Path to model trained without LoRA checkpoint
        model_lora_path: Path to model trained with LoRA checkpoint
        repo_dir: Path to DINOv3 repository directory
        weights: Path to frozen backbone weights
        seed: Random seed for reproducible sample selection
    """
    torch.manual_seed(seed)

    output_dir = Path(output_dir) / f"{dataset_name}_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running Segmentation Visualization for {dataset_name}")
    print(f"{'=' * 60}\n")

    # Get dataset info and create dataloader
    if dataset_name == "deepglobe":
        eval_transform = get_deepglobe_transforms(image_size, is_train=False)
        dataset = DeepGlobeDataset(
            data_root=data_root,
            split="test",
            transform=eval_transform,
            joint_transform=None,
            image_size=image_size,
        )
    elif dataset_name == "coralscapes":
        eval_transform = get_coralscapes_transforms(image_size, is_train=False)
        dataset = CoralscapesDataset(
            data_root=data_root,
            split="test",
            transform=eval_transform,
            joint_transform=None,
            image_size=image_size,
        )
    else:
        print(f"Segmentation visualization not supported for {dataset_name}")
        return

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Load models from provided paths
    models = {}
    exp_config = EXPERIMENTS[dataset_name]

    # Load model trained without LoRA if path provided
    if model_no_lora_path and Path(model_no_lora_path).exists():
        print(f"Loading model without LoRA from {model_no_lora_path}...")
        model_no_lora = create_model(
            model_type="dinov3",
            task="segmentation",
            freeze_backbone=True,
            num_classes=exp_config["num_classes"],
            use_lora=False,
            device=device,
            weights=weights,
            repo_dir=repo_dir
        )
        checkpoint = torch.load(model_no_lora_path, map_location=device)
        model_no_lora.load_state_dict(checkpoint["model_state_dict"])
        models["DINOv3 without LoRA"] = model_no_lora
    elif model_no_lora_path:
        print(f"Warning: Model without LoRA path provided but not found: {model_no_lora_path}")

    # Load model trained with LoRA if path provided
    if model_lora_path and Path(model_lora_path).exists():
        print(f"Loading model with LoRA from {model_lora_path}...")
        lora_config = {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["qkv", "proj", "fc1", "fc2"],
        }
        model_lora = create_model(
            model_type="dinov3",
            task="segmentation",
            freeze_backbone=True,
            num_classes=exp_config["num_classes"],
            use_lora=True,
            lora_config=lora_config,  # LoRA config will be loaded from state dict
            device=device,
            weights=weights,
            repo_dir=repo_dir
        )
        checkpoint = torch.load(model_lora_path, map_location=device)
        model_lora.load_state_dict(checkpoint["model_state_dict"])
        models["DINOv3 with LoRA"] = model_lora
    elif model_lora_path:
        print(f"Warning: Model with LoRA path provided but not found: {model_lora_path}")

    if not models:
        print("No models loaded. Provide at least one model path.")
        return

    # Move models to device and set to eval
    for model in models.values():
        model.to(device)
        model.eval()

    print(f"Comparing {len(models)} models: {list(models.keys())}")
    print(f"Using {num_samples} samples from {dataset_name}")

    # Segmentation map comparison
    print("\nGenerating segmentation map comparison...")
    figs = compare_segmentation_maps(
        models=models,
        dataloader=dataloader,
        num_samples=num_samples,
        device=device,
        save_dir=output_dir,
        figsize_per_image=(3.5, 3.5),
        dataset_name=dataset_name,
        with_overlay=overlay,
        overlay_alpha=overlay_alpha,
        num_classes=exp_config["num_classes"],
    )
    for fig in figs:
        plt.close(fig)

    print(f"\nSegmentation visualizations saved to: {output_dir}")


def run_all_segmentation_visualizations(
        trained_models_dir: str = "./outputs",
        data_root: Path = Path("./data"),
        output_dir: Path = Path("./visualizations"),
        device: str = "cuda",
        num_samples: int = 4,
        overlay: bool = False,
        overlay_alpha: float = 0.5,
        repo_dir: str = '',
        weights: str = '',
        seed: int = 42
) -> None:
    """
    Load trained segmentation models and run comparison visualizations.

    Looks for saved checkpoints in results_dir and compares:
    - Frozen DINOv3 backbone
    - Model trained without LoRA
    - Model trained with LoRA

    For each of 3 datasets (patchcamelyon, deepglobe, coralscapes), there are 2 models:
    - without LoRA
    - with LoRA

    Args:
        trained_models_dir: Directory containing trained model checkpoints
        data_root: Path to data directory
        output_dir: Path to output directory
        device: Device for computation
        num_samples: Number of samples to visualize
        overlay: Whether to create overlay visualizations
        overlay_alpha: Alpha value for overlay transparency
        repo_dir: Path to DINOv3 repository directory
        weights: Path to frozen backbone weights
        seed: Random seed for reproducible sample selection
    """
    print(f"\n{'=' * 60}")
    print("Running All Segmentation Visualizations")
    print(f"{'=' * 60}\n")

    common_end = 'checkpoint_best.pt'
    trained_models_paths = [
        f'{trained_models_dir}/coralscapes_dinov3_20251229_120012/{common_end}',  # coralscapes without LoRA
        f'{trained_models_dir}/coralscapes_dinov3_lora_20251229_205434/{common_end}',  # coralscapes with LoRA
        f'{trained_models_dir}/deepglobe_dinov3_20251229_224252/{common_end}',  # deepglobe without LoRA
        f'{trained_models_dir}/deepglobe_dinov3_lora_20251230_003334/{common_end}',  # deepglobe with LoRA
    ]

    segmentation_datasets = ["coralscapes", "deepglobe"]

    for dataset_idx, dataset_name in enumerate(segmentation_datasets):
        model_no_lora_path = trained_models_paths[dataset_idx * 2]
        model_lora_path = trained_models_paths[dataset_idx * 2 + 1]

        exp_config = EXPERIMENTS[dataset_name]

        run_segmentation_visualization(
            dataset_name=dataset_name,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
            num_samples=num_samples,
            image_size=exp_config["image_size_visualizations"],
            overlay=overlay,
            overlay_alpha=overlay_alpha,
            model_no_lora_path=model_no_lora_path,
            model_lora_path=model_lora_path,
            repo_dir=repo_dir,
            weights=weights,
            seed=seed,
        )

    print(f"\nAll segmentation visualizations complete.")


# ====================
# CLI Interface
# ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv3 Domain Shift Project"
    )

    parser.add_argument(
        "--action",
        type=str,
        choices=["download", "train", "visualize", "all"],
        default="all",
        help="Action to perform",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["patchcamelyon", "deepglobe", "coralscapes", "all"],
        default="all",
        help="Dataset to use",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["dinov3"],
        default="dinov3",
        help="Model to train (we only train DINOv3)",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Whether to freeze the backbone during training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA fine-tuning",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training",
    )

    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA fine-tuning",
    )

    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1 / 2,
        help="Weight for cross-entropy loss",
    )

    parser.add_argument(
        "--jaccard_weight",
        type=float,
        default=1 / 2,
        help="Weight for Jaccard loss",
    )

    parser.add_argument(
        "--dice_weight",
        type=float,
        default=0.0,
        help="Weight for Dice loss",
    )

    parser.add_argument(
        "--ignore_index",
        type=int,
        default=255,
        help="Index to ignore in segmentation loss",
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("./data"),
        help="Path to data directory",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs"),
        help="Path to output directory",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device for computation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=4,
        help="Number of samples for visualization",
    )

    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Whether to create overlay visualizations for segmentation",
    )

    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.5,
        help="Alpha value for overlay transparency",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Number of epochs without improvement before early stopping",
    )

    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-3,
        help="Minimum change in validation metric to qualify as an improvement",
    )

    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Whether to resume training from last checkpoint if available",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--epochs_resume_training",
        type=int,
        default=0,
        help="Target total number of epochs (including previous training) when resuming training",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default='',
        help="Path to frozen backbone weights for DINOv3",
    )

    parser.add_argument(
        "--repo_dir",
        type=str,
        default='',
        help="Path to the repository directory for custom DINOv3 code",
    )

    parser.add_argument(
        "--trained_models_dir",
        type=str,
        default="./outputs",
        help="Directory containing trained models' checkpoints",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"Action: {args.action}")
    print(f"Device: {args.device}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 60}\n")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.data_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "download":
        download_all_datasets(args.data_root)

    elif args.action == "train":
        if args.dataset == "all":
            run_all_experiments(
                data_root=args.data_root,
                output_dir=args.output_dir,
                freeze_backbone=args.freeze_backbone,
                ce_weight=args.ce_weight,
                jaccard_weight=args.jaccard_weight,
                dice_weight=args.dice_weight,
                ignore_index=args.ignore_index,
                device=args.device,
                seed=args.seed,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                weights=args.weights,
                repo_dir=args.repo_dir,
            )
        else:
            exp_config = EXPERIMENTS[args.dataset]
            exp_config["batch_size"] = args.batch_size
            exp_config["epochs"] = args.epochs
            exp_config["learning_rate"] = args.learning_rate
            exp_config["lora_learning_rate"] = args.lora_learning_rate
            run_experiment(
                dataset_name=args.dataset,
                model_type=args.model,
                freeze_backbone=args.freeze_backbone,
                use_lora=args.use_lora,
                ce_weight=args.ce_weight,
                jaccard_weight=args.jaccard_weight,
                dice_weight=args.dice_weight,
                ignore_index=args.ignore_index,
                data_root=args.data_root,
                output_dir=args.output_dir,
                device=args.device,
                seed=args.seed,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                resume_training=args.resume_training,
                checkpoint_path=Path(args.checkpoint_path),
                epochs_resume_training=args.epochs_resume_training,
                weights=args.weights,
                repo_dir=args.repo_dir,
            )

    elif args.action == "visualize":
        import matplotlib.pyplot as plt
        run_feature_visualization(
            data_root=args.data_root,
            output_dir=args.output_dir / "visualizations",
            device=args.device,
            num_samples=args.num_vis_samples,
            weights=args.weights,
            repo_dir=args.repo_dir,
            seed=args.seed
        )

        run_all_segmentation_visualizations(
            trained_models_dir=args.output_dir,
            data_root=args.data_root,
            output_dir=args.output_dir / "visualizations",
            device=args.device,
            num_samples=args.num_vis_samples,
            overlay=args.overlay,
            overlay_alpha=args.overlay_alpha,
            repo_dir=args.repo_dir,
            weights=args.weights,
            seed=args.seed,
        )

    elif args.action == "all":
        # Download datasets
        print("\n[1/3] Downloading datasets...")
        download_all_datasets(args.data_root)

        # Run all training experiments
        print("\n[2/3] Running training experiments...")
        run_all_experiments(
            data_root=args.data_root,
            output_dir=args.output_dir,
            freeze_backbone=args.freeze_backbone,
            ce_weight=args.ce_weight,
            jaccard_weight=args.jaccard_weight,
            dice_weight=args.dice_weight,
            ignore_index=args.ignore_index,
            device=args.device,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            weights=args.weights,
            repo_dir=args.repo_dir,
        )

        # Run visualizations
        print("\n[3/3] Running feature and segmentation visualizations...")
        import matplotlib.pyplot as plt
        run_feature_visualization(
            data_root=args.data_root,
            output_dir=args.output_dir / "visualizations",
            device=args.device,
            num_samples=args.num_vis_samples,
            weights=args.weights,
            repo_dir=args.repo_dir,
            seed=args.seed
        )

        run_all_segmentation_visualizations(
            trained_models_dir=args.output_dir,
            data_root=args.data_root,
            output_dir=args.output_dir / "visualizations",
            device=args.device,
            num_samples=args.num_vis_samples,
            overlay=args.overlay,
            overlay_alpha=args.overlay_alpha,
            repo_dir=args.repo_dir,
            weights=args.weights,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
