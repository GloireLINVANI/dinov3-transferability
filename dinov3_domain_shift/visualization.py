"""
Visualization Module
====================
PCA visualization and cosine similarity maps for dense feature analysis, and
segmentation maps comparison across models.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from torch import Tensor

from domain_shift_datasets import DEEPGLOBE_COLORS

# Set up the plotting style
plt.style.use('seaborn-v0_8-paper')

# Configure matplotlib for LaTeX compatibility and high-quality output
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.format': 'pdf',
})

# =================================
# Image Preprocessing Utilities
# =================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image(
        image: Union[Image.Image, np.ndarray, Tensor],
        image_size: int | Tuple[int, int] = 518,
        patch_size: int = 16,
) -> Tuple[Tensor, Tensor]:
    """
    Preprocess image for DINO models
    
    Returns:
        normalized_tensor: Normalized tensor for model input
        original_tensor: Original tensor for visualization
    """

    if isinstance(image_size, tuple):
        image_size = image_size[0]  # Use height for resizing
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, Tensor):
        image = TF.to_pil_image(image)

    # Resize to be divisible by patch size
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size
    # print("Resizing image to:", new_h, new_w)

    image_resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Convert to tensor
    original_tensor = TF.to_tensor(image_resized)

    # Normalize for the model
    normalized_tensor = TF.normalize(
        original_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )

    return normalized_tensor, original_tensor


def denormalize_image(tensor: Tensor) -> Tensor:
    """Denormalize ImageNet normalized tensor for visualization"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


# ========================
# Feature Extraction
# ========================

@torch.no_grad()
def extract_dense_features(
        model: torch.nn.Module,
        images: Tensor,
        num_layers: int = 24,
) -> Tensor:
    """
    Extract dense patch features from DINO model
    
    Args:
        model: DINO model
        images: [B, C, H, W] input images
        num_layers: Number of layers in the model
    
    Returns:
        features: [B, D, H_patches, W_patches] dense features
    """
    model.eval()
    features = model.get_intermediate_layers(
        images, n=[num_layers - 1], reshape=True, norm=True
    )
    return features[0]


# ======================================
# PCA Visualization
# ======================================

class PCAVisualizer:
    """
    PCA-based feature visualization
    Projects dense features to RGB using PCA
    """

    def __init__(
            self,
            n_components: int = 3,
            whiten: bool = True,
            sigmoid_scale: float = 2.0,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.sigmoid_scale = sigmoid_scale
        self.pca = None

    def fit(self, features: Tensor, mask: Optional[Tensor] = None):
        """
        Fit PCA on features
        
        Args:
            features: [B, D, H, W] or [N, D] features
            mask: Optional [B, H, W] or [N] boolean mask for foreground
        """
        # Reshape features to [N, D]
        if features.dim() == 4:
            B, D, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, D)
            if mask is not None:
                mask = mask.reshape(-1)

        features = features.cpu().numpy()

        # Apply mask if provided
        if mask is not None:
            mask = mask.cpu().numpy().astype(bool)
            features = features[mask]

        # Fit PCA
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(features)

        return self

    def transform(self, features: Tensor) -> Tensor:
        """
        Transform features to RGB using fitted PCA
        
        Args:
            features: [B, D, H, W] or [N, D] features
        
        Returns:
            rgb: [B, 3, H, W] or [N, 3] RGB values
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        is_4d = features.dim() == 4

        if is_4d:
            B, D, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, D)

        features = features.cpu().numpy()

        # Project using PCA
        projected = self.pca.transform(features)
        projected = torch.from_numpy(projected).float()

        # Apply sigmoid with scaling for vibrant colors
        rgb = torch.sigmoid(projected * self.sigmoid_scale)

        if is_4d:
            rgb = rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return rgb

    def fit_transform(
            self, features: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Fit PCA and transform in one step"""
        self.fit(features, mask)
        return self.transform(features)


# ==========================
# Cosine Similarity Maps
# ==========================

def compute_cosine_similarity_map(
        features: Tensor,
        query_point: Tuple[int, int],
) -> Tensor:
    """
    Compute cosine similarity between a query point and all other patches
    
    Args:
        features: [D, H, W] dense features for one image
        query_point: (row, col) patch coordinates
    
    Returns:
        similarity_map: [H, W] cosine similarity values
    """
    D, H, W = features.shape

    # Get query feature
    row, col = query_point
    query_feature = features[:, row, col]  # [D]

    # Normalize features
    features_flat = features.view(D, -1)  # [D, H*W]
    features_norm = F.normalize(features_flat, p=2, dim=0)
    query_norm = F.normalize(query_feature.unsqueeze(1), p=2, dim=0)

    # Compute cosine similarity
    similarity = torch.mm(query_norm.T, features_norm)  # [1, H*W]
    similarity_map = similarity.view(H, W)

    return similarity_map


# =================================
# Model Comparison Visualization
# =================================

def compare_models_pca(
        models: dict,
        images: Union[List[Image.Image], Image.Image, Tensor],
        num_layers: dict,
        patch_sizes: dict,
        image_size: int | Tuple[int, int] = 512,
        device: str = "cuda",
        save_path: Optional[Path] = None,
        dataset_name: str = "",
        idx: int | None = None,
) -> plt.Figure:
    """
    Compare PCA visualizations across multiple models (DINOv2 vs DINOv3)
    
    Args:
        models: Dict of model name to model
        images: Input image(s)
        num_layers: Dict of model name to number of layers
        patch_sizes: Dict of model name to patch size
        image_size: Target image size (height, width adjusted to preserve aspect ratio)
        device: Device for computation
        save_path: Optional path to save figure
        dataset_name: Optional dataset name
        idx: Image index (only used for Coralscapes)
    
    Returns:
        matplotlib Figure
    """
    if isinstance(images, Image.Image) or isinstance(images, Tensor):
        images = [images]

    n_models = len(models)
    n_images = len(images)

    fig, axes = plt.subplots(
        n_images, n_models + 1,
        figsize=(4 * (n_models + 1), 4 * n_images),
        constrained_layout=True,
        gridspec_kw={'hspace': 0.02, 'wspace': 0.02}
    )

    if n_images == 1:
        axes = axes.reshape(1, -1)

    # Preprocess images for each model (image sizes may differ due to patch size)
    for img_idx, image in enumerate(images):
        # Show original image
        # if isinstance(image, Tensor) and dataset_name == "patchcamelyon":
        #    orig_tensor = denormalize_image(image)  # We assume image is already normalized and has the correct size
        # else:
        if isinstance(image_size, int):
            resize_size = (image_size, image_size)
        else:
            # image_size is (height, width), but PIL.resize expects (width, height)
            resize_size = (image_size[1], image_size[0])
        orig_tensor = TF.to_tensor(image.resize(resize_size))
        axes[img_idx, 0].imshow(orig_tensor.permute(1, 2, 0))
        if img_idx == 0:
            if dataset_name != "coralscapes" or idx == 0:
                axes[img_idx, 0].set_title(f"Input", fontsize=10)
        axes[img_idx, 0].axis('off')

        # Process each model
        for model_idx, (model_name, model) in enumerate(models.items()):
            model = model.to(device)
            model.eval()

            patch_size = patch_sizes[model_name]
            n_layers = num_layers[model_name]

            # Preprocess
            # if dataset_name == "patchcamelyon":
            # images_tensor = image.unsqueeze(0).to(device)
            # else:
            norm_tensor, _ = preprocess_image(image, image_size, patch_size)
            images_tensor = norm_tensor.unsqueeze(0).to(device)

            # Extract features
            with torch.no_grad():
                features = extract_dense_features(model, images_tensor, n_layers)

            # PCA visualization
            pca_viz = PCAVisualizer(n_components=3, whiten=True, sigmoid_scale=2.0)
            rgb_features = pca_viz.fit_transform(features)

            # Plot
            pca_img = rgb_features[0].permute(1, 2, 0).cpu()
            axes[img_idx, model_idx + 1].imshow(pca_img)

            if img_idx == 0:
                if dataset_name != "coralscapes" or idx == 0:
                    axes[img_idx, model_idx + 1].set_title(f"{model_name}", fontsize=10)
            axes[img_idx, model_idx + 1].axis('off')

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved model comparison to {save_path}")

    return fig


def compare_models_cosine_similarity(
        models: dict,
        images: Union[List[Image.Image], Image.Image, List[Tensor], Tensor],
        query_points: Union[List[Tuple[int, int]], Tuple[int, int], None] = None,
        num_layers: dict = None,
        patch_sizes: dict = None,
        image_size: int | Tuple[int, int] = 512,
        device: str = "cuda",
        save_path: Optional[Path] = None,
        dataset_name: str = "",
        idx: int | None = None,
) -> plt.Figure:
    """
    Compare cosine similarity maps across multiple models for one or more images

    Args:
        models: Dict of model name to model
        images: Input image(s) - can be a single image or list of images
        query_points: Query point(s) (row, col) - can be a single point, list of points, or None.
                     If None, defaults to center. If single point, used for all images.
                     If list, should match the number of images.
        num_layers: Dict of model name to number of layers
        patch_sizes: Dict of model name to patch size
        image_size: Target image size
        device: Device for computation
        save_path: Optional path to save figure
        dataset_name: Optional dataset name
        idx: Image index (only used for Coralscapes)

    Returns:
        matplotlib Figure
    """
    # Convert single image to list
    if isinstance(images, (Image.Image, Tensor)):
        images = [images]

    # Handle query points
    if query_points is None:
        query_points = [None] * len(images)
    elif isinstance(query_points, tuple):
        # Single query point, we use it for all images
        query_points = [query_points] * len(images)

    n_models = len(models)
    n_images = len(images)

    fig, axes = plt.subplots(
        n_images, n_models + 1,
        figsize=(4 * (n_models + 1), 4 * n_images),
        constrained_layout=True,
        gridspec_kw={'hspace': 0.02, 'wspace': 0.02}
    )

    # Handle single row case
    if n_images == 1:
        axes = axes.reshape(1, -1)

    # Process each image
    for img_idx, (image, query_point) in enumerate(zip(images, query_points)):
        # Show original image
        # if isinstance(image, Tensor) and dataset_name == "patchcamelyon":
        #    orig_tensor = denormalize_image(image)
        # else:
        if isinstance(image_size, int):
            resize_size = (image_size, image_size)
        else:
            # image_size is (height, width), but PIL.resize expects (width, height)
            resize_size = (image_size[1], image_size[0])
        orig_tensor = TF.to_tensor(image.resize(resize_size))

        axes[img_idx, 0].imshow(orig_tensor.permute(1, 2, 0))
        if img_idx == 0:
            if dataset_name != "coralscapes" or idx == 0:
                axes[img_idx, 0].set_title(f"Input", fontsize=10)
        axes[img_idx, 0].axis('off')

        # Process each model
        for model_idx, (model_name, model) in enumerate(models.items()):
            model = model.to(device)
            model.eval()

            patch_size = patch_sizes[model_name]
            n_layers = num_layers[model_name]

            # Preprocess
            # if dataset_name == "patchcamelyon":
            #    images_tensor = image.unsqueeze(0).to(device)
            # else:
            norm_tensor, _ = preprocess_image(image, image_size, patch_size)
            images_tensor = norm_tensor.unsqueeze(0).to(device)

            n_patches_h = images_tensor.size(2) // patch_size
            n_patches_w = images_tensor.size(3) // patch_size
            if query_point is None:
                # Default to center patch
                current_query_point = (n_patches_h // 2, n_patches_w // 2)
            else:
                current_query_point = query_point

            # Adjust query point for this model's patch size
            row = min(current_query_point[0], n_patches_h - 1)
            col = min(current_query_point[1], n_patches_w - 1)

            # Extract features
            with torch.no_grad():
                features = extract_dense_features(model, images_tensor, n_layers)

            features = features[0].cpu()

            # Compute similarity map
            similarity_map = compute_cosine_similarity_map(features, (row, col))

            # Plot
            im = axes[img_idx, model_idx + 1].imshow(
                similarity_map.numpy(), cmap='viridis', vmin=-1, vmax=1
            )
            axes[img_idx, model_idx + 1].scatter(
                col, row, c='red', s=50, marker='x', linewidths=2
            )
            if img_idx == 0:
                if dataset_name != "coralscapes" or idx == 0:
                    axes[img_idx, model_idx + 1].set_title(f"{model_name}", fontsize=10)
            axes[img_idx, model_idx + 1].axis('off')
            if model_idx == n_models - 1:
                plt.colorbar(im, ax=axes[img_idx, model_idx + 1], fraction=0.046, pad=0.04)

            # Mark on original (only once, using the first model's patch size)
            if model_idx == 0:
                pixel_y = (row + 0.5) * patch_size
                pixel_x = (col + 0.5) * patch_size
                axes[img_idx, 0].scatter(
                    pixel_x, pixel_y, c='red', s=100, marker='x', linewidths=2
                )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved model comparison to {save_path}")

    return fig


# ===================================
# Batch Visualization for a dataset
# ===================================

def visualize_dataset_features(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 8,
        num_layers: int = 24,
        device: str = "cuda",
        save_dir: Optional[Path] = None,
        model_name: str = "model",
) -> List[plt.Figure]:
    """
    Visualize features for samples from a dataset
    
    Args:
        model: DINO model
        dataloader: DataLoader for the dataset
        num_samples: Number of samples to visualize
        num_layers: Number of layers in model
        device: Device for computation
        save_dir: Optional directory to save figures
        model_name: Name for saving
    
    Returns:
        List of matplotlib figures
    """
    model = model.to(device)
    model.eval()

    figures = []
    samples_visualized = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        if samples_visualized >= num_samples:
            break

        for i in range(min(len(images), num_samples - samples_visualized)):
            image_tensor = images[i:i + 1].to(device)

            # Extract features
            with torch.no_grad():
                features = extract_dense_features(model, image_tensor, num_layers)

            # PCA visualization
            pca_viz = PCAVisualizer()
            rgb_features = pca_viz.fit_transform(features)

            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            # Original
            orig = denormalize_image(images[i].cpu()).permute(1, 2, 0).clamp(0, 1)
            axes[0].imshow(orig)
            axes[0].set_title("Original", fontsize=10)
            axes[0].axis('off')

            # PCA
            pca_img = rgb_features[0].permute(1, 2, 0).cpu()
            axes[1].imshow(pca_img)
            axes[1].set_title("PCA Features", fontsize=10)
            axes[1].axis('off')

            plt.tight_layout()

            if save_dir:
                save_path = Path(save_dir) / f"{model_name}_sample_{samples_visualized}.pdf"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, bbox_inches='tight')

            figures.append(fig)
            samples_visualized += 1

    return figures


# ========================================
# Segmentation Comparison Visualization
# ========================================

def get_segmentation_colormap(num_classes: int, include_background: bool = False,
                              dataset_name: str = "dataset") -> np.ndarray:
    """
    Generate a colormap for segmentation visualization.

    Args:
        num_classes: Number of classes
        include_background: Whether class 0 is background (will be black)
        dataset_name: Name of the dataset ("deepglobe", etc.) to use dataset-specific colors

    Returns:
        numpy array of shape [num_classes, 3] with RGB values in [0, 1]
    """
    # Use DeepGlobe colors if dataset is "deepglobe"
    if dataset_name == "deepglobe":
        cmap = np.zeros((num_classes, 3), dtype=np.float32)
        # Convert DEEPGLOBE_COLORS dict {(R, G, B): class_idx} to array [num_classes, 3]
        for rgb_tuple, class_idx in DEEPGLOBE_COLORS.items():
            if class_idx < num_classes:
                cmap[class_idx] = np.array(rgb_tuple, dtype=np.float32) / 255.0
        return cmap

    if num_classes <= 21:
        # PASCAL VOC colormap for small number of classes
        cmap = np.array([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ], dtype=np.float32) / 255.0
        return cmap[:num_classes]
    else:
        # Generate colormap using HSV space for many classes
        cmap = np.zeros((num_classes, 3), dtype=np.float32)
        if include_background:
            cmap[0] = [0, 0, 0]  # Background is black
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, num_classes):
            hue = (i - start_idx) / (num_classes - start_idx)
            # Convert HSV to RGB
            rgb = mcolors.hsv_to_rgb([hue, 0.8, 0.9])
            cmap[i] = rgb

        return cmap


def mask_to_rgb(
        mask: np.ndarray,
        num_classes: int,
        colormap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert segmentation mask to RGB image.

    Args:
        mask: [H, W] integer mask with class indices
        num_classes: Number of classes
        colormap: Optional [num_classes, 3] colormap array

    Returns:
        [H, W, 3] RGB image
    """
    if colormap is None:
        colormap = get_segmentation_colormap(num_classes)

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for c in range(num_classes):
        rgb[mask == c] = colormap[c]

    return rgb


def compare_segmentation_maps(
        models: dict,  # {"name": model}
        dataloader,
        num_classes: int,
        num_samples: int = 4,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
        save_dir: Optional[Path] = None,
        figsize_per_image: Tuple[float, float] = (3, 3),
        dataset_name: str = "dataset",
        with_overlay: bool = False,
        overlay_alpha: float = 0.5,
) -> List[plt.Figure]:
    """
    Compare segmentation predictions from multiple models.

    Creates a grid visualization with:
    - Columns: Original image, Ground truth, Model predictions
    - Rows: Different samples

    Args:
        models: Dict mapping model names to model objects
        dataloader: DataLoader yielding (images, masks) tuples
        num_classes: Number of segmentation classes
        num_samples: Number of samples to visualize
        device: Device for computation
        class_names: Optional list of class names for legend
        save_dir: Optional directory to save figures
        figsize_per_image: Figure size per image cell
        dataset_name: Name of the dataset ("deepglobe", "coralscapes", etc.) for dataset-specific colors.
                     If "coralscapes", saves each sample as a separate PDF file.
        with_overlay: If True, also generate and save overlay visualizations
        overlay_alpha: Transparency for overlay (0=image only, 1=mask only)

    Returns:
        List of matplotlib Figures (main figure + overlay figure if with_overlay=True)
    """
    # Generate colormap
    colormap = get_segmentation_colormap(num_classes, dataset_name=dataset_name)

    for model in models.values():
        model.to(device)
        model.eval()

    figures = []
    samples_collected = 0

    # Collect samples
    all_images = []
    all_masks = []

    for images, masks in dataloader:
        for i in range(len(images)):
            if samples_collected >= num_samples:
                break
            all_images.append(images[i])
            all_masks.append(masks[i])
            samples_collected += 1
        if samples_collected >= num_samples:
            break

    n_models = len(models)
    n_rows = len(all_images)
    n_cols = 2 + n_models  # Original + GT + models

    # Column labels
    column_labels = ["Input", "Ground Truth"] + list(models.keys())

    # Store predictions for potential overlay figure
    all_predictions = []

    # If coralscapes, create separate figures for each sample
    if dataset_name == "coralscapes":
        for sample_idx, (image, gt_mask) in enumerate(zip(all_images, all_masks)):
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1]),
            )

            # Denormalize image for display
            img_display = denormalize_image(image).permute(1, 2, 0).clamp(0, 1).numpy()

            # Convert ground truth mask to RGB
            gt_mask_np = gt_mask.numpy()
            gt_rgb = mask_to_rgb(gt_mask_np, num_classes, colormap)

            # Column 0: Original image
            axes[0].imshow(img_display)
            axes[0].axis('off')
            # Column 1: Ground truth
            axes[1].imshow(gt_rgb)
            axes[1].axis('off')
            if sample_idx == 0:
                axes[0].set_title(column_labels[0], fontsize=10, fontweight='bold')
                axes[1].set_title(column_labels[1], fontsize=10, fontweight='bold')

            # Column 2+: Model predictions
            predictions_for_row = [(img_display, gt_rgb)]  # Store for overlay
            for model_idx, (model_name, model) in enumerate(models.items()):
                with torch.no_grad():
                    image_input = image.unsqueeze(0).to(device)
                    logits = model(image_input)
                    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

                pred_rgb = mask_to_rgb(pred_mask, num_classes, colormap)
                predictions_for_row.append(pred_rgb)

                column = 2 + model_idx
                axes[column].imshow(pred_rgb)
                axes[column].axis('off')
                if sample_idx == 0:
                    axes[column].set_title(column_labels[column], fontsize=10, fontweight='bold')

            all_predictions.append(predictions_for_row)

            plt.subplots_adjust(hspace=0.05, wspace=0.05)

            if save_dir:
                save_path = Path(save_dir) / f"segmentation_comparison_sample_{sample_idx}.pdf"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path)
                print(f"Saved segmentation comparison for sample {sample_idx} to {save_path}")

            figures.append(fig)

        # Create overlay figures if requested
        if with_overlay:
            for sample_idx, predictions_row in enumerate(all_predictions):
                fig_overlay, axes_overlay = plt.subplots(
                    1, n_cols,
                    figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1]),
                )

                img_display, gt_rgb = predictions_row[0]

                # Column 0: Original image
                axes_overlay[0].imshow(img_display)
                axes_overlay[0].axis('off')
                # Column 1: Ground truth overlay
                gt_overlay = (1 - overlay_alpha) * img_display + overlay_alpha * gt_rgb
                axes_overlay[1].imshow(gt_overlay)
                axes_overlay[1].axis('off')
                if sample_idx == 0:
                    axes_overlay[0].set_title(column_labels[0], fontsize=10, fontweight='bold')
                    axes_overlay[1].set_title(column_labels[1], fontsize=10, fontweight='bold')

                # Column 2+: Model prediction overlays
                for model_idx, pred_rgb in enumerate(predictions_row[1:]):
                    pred_overlay = (1 - overlay_alpha) * img_display + overlay_alpha * pred_rgb

                    column = 2 + model_idx
                    axes_overlay[column].imshow(pred_overlay)
                    axes_overlay[column].axis('off')
                    if sample_idx == 0:
                        axes_overlay[column].set_title(column_labels[column], fontsize=10, fontweight='bold')

                plt.subplots_adjust(hspace=0.05, wspace=0.05)

                if save_dir:
                    save_path_overlay = Path(save_dir) / f"segmentation_comparison_overlay_sample_{sample_idx}.pdf"
                    save_path_overlay.parent.mkdir(parents=True, exist_ok=True)
                    fig_overlay.savefig(save_path_overlay)
                    print(f"Saved segmentation overlay comparison for sample {sample_idx} to {save_path_overlay}")

                figures.append(fig_overlay)

    else:
        # Create main (non-overlay) figure
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows),
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row_idx, (image, gt_mask) in enumerate(zip(all_images, all_masks)):
            # Denormalize image for display
            img_display = denormalize_image(image).permute(1, 2, 0).clamp(0, 1).numpy()

            # Convert ground truth mask to RGB
            gt_mask_np = gt_mask.numpy()
            gt_rgb = mask_to_rgb(gt_mask_np, num_classes, colormap)

            # Column 0: Original image
            axes[row_idx, 0].imshow(img_display)
            axes[row_idx, 0].axis('off')
            if row_idx == 0:
                axes[row_idx, 0].set_title(column_labels[0], fontsize=10, fontweight='bold')

            # Column 1: Ground truth
            axes[row_idx, 1].imshow(gt_rgb)
            axes[row_idx, 1].axis('off')
            if row_idx == 0:
                axes[row_idx, 1].set_title(column_labels[1], fontsize=10, fontweight='bold')

            # Column 2+: Model predictions
            predictions_for_row = [(img_display, gt_rgb)]  # Store for overlay
            for model_idx, (model_name, model) in enumerate(models.items()):
                with torch.no_grad():
                    image_input = image.unsqueeze(0).to(device)
                    logits = model(image_input)
                    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

                pred_rgb = mask_to_rgb(pred_mask, num_classes, colormap)
                predictions_for_row.append(pred_rgb)

                column = 2 + model_idx
                axes[row_idx, column].imshow(pred_rgb)
                axes[row_idx, column].axis('off')
                if row_idx == 0:
                    axes[row_idx, column].set_title(column_labels[column], fontsize=10, fontweight='bold')

            all_predictions.append(predictions_for_row)

        plt.subplots_adjust(hspace=0.05, wspace=0.05)

        if save_dir:
            save_path = Path(save_dir) / "segmentation_comparison.pdf"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"Saved segmentation comparison to {save_path}")

        figures.append(fig)

        # Create overlay figure if requested
        if with_overlay:
            fig_overlay, axes_overlay = plt.subplots(
                n_rows, n_cols,
                figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows),
            )

            if n_rows == 1:
                axes_overlay = axes_overlay.reshape(1, -1)

            for row_idx, predictions_row in enumerate(all_predictions):
                img_display, gt_rgb = predictions_row[0]

                # Column 0: Original image
                axes_overlay[row_idx, 0].imshow(img_display)
                axes_overlay[row_idx, 0].axis('off')
                if row_idx == 0:
                    axes_overlay[row_idx, 0].set_title(column_labels[0], fontsize=10, fontweight='bold')

                # Column 1: Ground truth overlay
                gt_overlay = (1 - overlay_alpha) * img_display + overlay_alpha * gt_rgb
                axes_overlay[row_idx, 1].imshow(gt_overlay)
                axes_overlay[row_idx, 1].axis('off')
                if row_idx == 0:
                    axes_overlay[row_idx, 1].set_title(column_labels[1], fontsize=10, fontweight='bold')

                # Column 2+: Model prediction overlays
                for model_idx, pred_rgb in enumerate(predictions_row[1:]):
                    pred_overlay = (1 - overlay_alpha) * img_display + overlay_alpha * pred_rgb

                    column = 2 + model_idx
                    axes_overlay[row_idx, column].imshow(pred_overlay)
                    axes_overlay[row_idx, column].axis('off')
                    if row_idx == 0:
                        axes_overlay[row_idx, column].set_title(column_labels[column], fontsize=10,
                                                                fontweight='bold')

            plt.subplots_adjust(hspace=0.05, wspace=0.05)

            if save_dir:
                save_path_overlay = Path(save_dir) / "segmentation_comparison_overlay.pdf"
                save_path_overlay.parent.mkdir(parents=True, exist_ok=True)
                fig_overlay.savefig(save_path_overlay)
                print(f"Saved segmentation overlay comparison to {save_path_overlay}")

            figures.append(fig_overlay)

    return figures


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing visualization utilities...")

    # Create dummy features
    features = torch.randn(1, 1024, 32, 32)

    # Test PCA visualizer
    pca_viz = PCAVisualizer()
    rgb = pca_viz.fit_transform(features)
    print(f"PCA output shape: {rgb.shape}")

    # Test cosine similarity
    sim_map = compute_cosine_similarity_map(features[0], (16, 16))
    print(f"Similarity map shape: {sim_map.shape}")
    print(sim_map)
