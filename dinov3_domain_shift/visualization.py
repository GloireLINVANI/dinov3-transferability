"""PCA and cosine similarity maps of dense features, and comparisons of segmentation maps across models."""

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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image(
        image: Union[Image.Image, np.ndarray, Tensor],
        image_size: int | Tuple[int, int] = 518,
        patch_size: int = 16,
) -> Tuple[Tensor, Tensor]:
    """Return the image resized for the patch size, normalized for the model and unnormalized for display."""

    if isinstance(image_size, tuple):
        image_size = image_size[0]  # Use height for resizing
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

    image_resized = image.resize((new_w, new_h), Image.BILINEAR)

    original_tensor = TF.to_tensor(image_resized)

    normalized_tensor = TF.normalize(
        original_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )

    return normalized_tensor, original_tensor


def denormalize_image(tensor: Tensor) -> Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


@torch.no_grad()
def extract_dense_features(
        model: torch.nn.Module,
        images: Tensor,
        num_layers: int = 24,
) -> Tensor:
    """Return [B, D, H_patches, W_patches] dense patch features from a DINO model."""
    model.eval()
    features = model.get_intermediate_layers(
        images, n=[num_layers - 1], reshape=True, norm=True
    )
    return features[0]


class PCAVisualizer:
    """Project dense features to RGB colors with PCA."""

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
        """Fit the PCA on [B, D, H, W] or [N, D] features, optionally only where the boolean mask is True."""
        if features.dim() == 4:
            B, D, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, D)
            if mask is not None:
                mask = mask.reshape(-1)

        features = features.cpu().numpy()

        if mask is not None:
            mask = mask.cpu().numpy().astype(bool)
            features = features[mask]

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(features)

        return self

    def transform(self, features: Tensor) -> Tensor:
        """Map [B, D, H, W] or [N, D] features to [B, 3, H, W] or [N, 3] RGB values."""
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        is_4d = features.dim() == 4

        if is_4d:
            B, D, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, D)

        features = features.cpu().numpy()

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
        self.fit(features, mask)
        return self.transform(features)


def compute_cosine_similarity_map(
        features: Tensor,
        query_point: Tuple[int, int],
) -> Tensor:
    """Cosine similarity [H, W] between the query patch (row, col) and all patches of [D, H, W] features."""
    D, H, W = features.shape

    row, col = query_point
    query_feature = features[:, row, col]  # [D]

    features_flat = features.view(D, -1)  # [D, H*W]
    features_norm = F.normalize(features_flat, p=2, dim=0)
    query_norm = F.normalize(query_feature.unsqueeze(1), p=2, dim=0)

    similarity = torch.mm(query_norm.T, features_norm)  # [1, H*W]
    similarity_map = similarity.view(H, W)

    return similarity_map


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
    """Plot the PCA projections of each model's features side by side, for one or more images."""
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

        for model_idx, (model_name, model) in enumerate(models.items()):
            model = model.to(device)
            model.eval()

            patch_size = patch_sizes[model_name]
            n_layers = num_layers[model_name]

            norm_tensor, _ = preprocess_image(image, image_size, patch_size)
            images_tensor = norm_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                features = extract_dense_features(model, images_tensor, n_layers)

            pca_viz = PCAVisualizer(n_components=3, whiten=True, sigmoid_scale=2.0)
            rgb_features = pca_viz.fit_transform(features)

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
    """Plot each model's cosine similarity maps; query_points is None (center), one point, or one per image."""
    if isinstance(images, (Image.Image, Tensor)):
        images = [images]

    if query_points is None:
        query_points = [None] * len(images)
    elif isinstance(query_points, tuple):
        query_points = [query_points] * len(images)

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

    for img_idx, (image, query_point) in enumerate(zip(images, query_points)):
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

        for model_idx, (model_name, model) in enumerate(models.items()):
            model = model.to(device)
            model.eval()

            patch_size = patch_sizes[model_name]
            n_layers = num_layers[model_name]

            norm_tensor, _ = preprocess_image(image, image_size, patch_size)
            images_tensor = norm_tensor.unsqueeze(0).to(device)

            n_patches_h = images_tensor.size(2) // patch_size
            n_patches_w = images_tensor.size(3) // patch_size
            if query_point is None:
                current_query_point = (n_patches_h // 2, n_patches_w // 2)
            else:
                current_query_point = query_point

            # Adjust query point for this model's patch size
            row = min(current_query_point[0], n_patches_h - 1)
            col = min(current_query_point[1], n_patches_w - 1)

            with torch.no_grad():
                features = extract_dense_features(model, images_tensor, n_layers)

            features = features[0].cpu()

            similarity_map = compute_cosine_similarity_map(features, (row, col))

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


def visualize_dataset_features(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 8,
        num_layers: int = 24,
        device: str = "cuda",
        save_dir: Optional[Path] = None,
        model_name: str = "model",
) -> List[plt.Figure]:
    """Plot the PCA projection of the features of a few dataset samples."""
    model = model.to(device)
    model.eval()

    figures = []
    samples_visualized = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        if samples_visualized >= num_samples:
            break

        for i in range(min(len(images), num_samples - samples_visualized)):
            image_tensor = images[i:i + 1].to(device)

            with torch.no_grad():
                features = extract_dense_features(model, image_tensor, num_layers)

            pca_viz = PCAVisualizer()
            rgb_features = pca_viz.fit_transform(features)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            orig = denormalize_image(images[i].cpu()).permute(1, 2, 0).clamp(0, 1)
            axes[0].imshow(orig)
            axes[0].set_title("Original", fontsize=10)
            axes[0].axis('off')

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


def get_segmentation_colormap(num_classes: int, include_background: bool = False,
                              dataset_name: str = "dataset") -> np.ndarray:
    """Return a [num_classes, 3] RGB colormap in [0, 1], with DeepGlobe's own colors for that dataset."""
    if dataset_name == "deepglobe":
        cmap = np.zeros((num_classes, 3), dtype=np.float32)
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
        cmap = np.zeros((num_classes, 3), dtype=np.float32)
        if include_background:
            cmap[0] = [0, 0, 0]
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, num_classes):
            hue = (i - start_idx) / (num_classes - start_idx)
            rgb = mcolors.hsv_to_rgb([hue, 0.8, 0.9])
            cmap[i] = rgb

        return cmap


def mask_to_rgb(
        mask: np.ndarray,
        num_classes: int,
        colormap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert an [H, W] class-index mask to an [H, W, 3] RGB image."""
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
    """Plot each model's predicted segmentation maps next to the input and the ground truth, optionally as overlays."""
    colormap = get_segmentation_colormap(num_classes, dataset_name=dataset_name)

    for model in models.values():
        model.to(device)
        model.eval()

    figures = []
    samples_collected = 0

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

    column_labels = ["Input", "Ground Truth"] + list(models.keys())

    # Store predictions for potential overlay figure
    all_predictions = []

    if dataset_name == "coralscapes":
        for sample_idx, (image, gt_mask) in enumerate(zip(all_images, all_masks)):
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1]),
            )

            img_display = denormalize_image(image).permute(1, 2, 0).clamp(0, 1).numpy()

            gt_mask_np = gt_mask.numpy()
            gt_rgb = mask_to_rgb(gt_mask_np, num_classes, colormap)

            axes[0].imshow(img_display)
            axes[0].axis('off')
            axes[1].imshow(gt_rgb)
            axes[1].axis('off')
            if sample_idx == 0:
                axes[0].set_title(column_labels[0], fontsize=10, fontweight='bold')
                axes[1].set_title(column_labels[1], fontsize=10, fontweight='bold')

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

        if with_overlay:
            for sample_idx, predictions_row in enumerate(all_predictions):
                fig_overlay, axes_overlay = plt.subplots(
                    1, n_cols,
                    figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1]),
                )

                img_display, gt_rgb = predictions_row[0]

                axes_overlay[0].imshow(img_display)
                axes_overlay[0].axis('off')
                gt_overlay = (1 - overlay_alpha) * img_display + overlay_alpha * gt_rgb
                axes_overlay[1].imshow(gt_overlay)
                axes_overlay[1].axis('off')
                if sample_idx == 0:
                    axes_overlay[0].set_title(column_labels[0], fontsize=10, fontweight='bold')
                    axes_overlay[1].set_title(column_labels[1], fontsize=10, fontweight='bold')

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
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows),
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row_idx, (image, gt_mask) in enumerate(zip(all_images, all_masks)):
            img_display = denormalize_image(image).permute(1, 2, 0).clamp(0, 1).numpy()

            gt_mask_np = gt_mask.numpy()
            gt_rgb = mask_to_rgb(gt_mask_np, num_classes, colormap)

            axes[row_idx, 0].imshow(img_display)
            axes[row_idx, 0].axis('off')
            if row_idx == 0:
                axes[row_idx, 0].set_title(column_labels[0], fontsize=10, fontweight='bold')

            axes[row_idx, 1].imshow(gt_rgb)
            axes[row_idx, 1].axis('off')
            if row_idx == 0:
                axes[row_idx, 1].set_title(column_labels[1], fontsize=10, fontweight='bold')

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

        if with_overlay:
            fig_overlay, axes_overlay = plt.subplots(
                n_rows, n_cols,
                figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows),
            )

            if n_rows == 1:
                axes_overlay = axes_overlay.reshape(1, -1)

            for row_idx, predictions_row in enumerate(all_predictions):
                img_display, gt_rgb = predictions_row[0]

                axes_overlay[row_idx, 0].imshow(img_display)
                axes_overlay[row_idx, 0].axis('off')
                if row_idx == 0:
                    axes_overlay[row_idx, 0].set_title(column_labels[0], fontsize=10, fontweight='bold')

                gt_overlay = (1 - overlay_alpha) * img_display + overlay_alpha * gt_rgb
                axes_overlay[row_idx, 1].imshow(gt_overlay)
                axes_overlay[row_idx, 1].axis('off')
                if row_idx == 0:
                    axes_overlay[row_idx, 1].set_title(column_labels[1], fontsize=10, fontweight='bold')

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
    print("Testing visualization utilities...")

    features = torch.randn(1, 1024, 32, 32)

    pca_viz = PCAVisualizer()
    rgb = pca_viz.fit_transform(features)
    print(f"PCA output shape: {rgb.shape}")

    sim_map = compute_cosine_similarity_map(features[0], (16, 16))
    print(f"Similarity map shape: {sim_map.shape}")
    print(sim_map)
