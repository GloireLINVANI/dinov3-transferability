"""
Models Module
=============
DINOv2 and DINOv3 backbones with task-specific heads and LoRA fine-tuning
"""

import math
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================
# LoRA Implementation
# =========================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 16,
            alpha: int = 32,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        """Compute LoRA delta: scaling * (x @ A^T @ B^T)"""
        return self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(
            self,
            linear: nn.Linear,
            r: int = 16,
            alpha: int = 32,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features

        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(
        model: nn.Module,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: List[str] = ["qkv", "proj", "fc1", "fc2"],
) -> nn.Module:
    """Apply LoRA to specified modules in a model"""

    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        should_apply = any(target in name for target in target_modules)

        if should_apply and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRA version
            lora_linear = LinearWithLoRA(
                module, r=r, alpha=alpha, dropout=dropout
            )
            setattr(parent, attr_name, lora_linear)

    return model


# =================
# Model Loading
# =================

def load_dinov2(model_name: str = "dinov2_vitl14", pretrained: bool = True) -> nn.Module:
    """Load DINOv2 model from torch.hub"""
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
    model.eval()
    return model


def load_dinov3(
        model_name: str = "dinov3_vitl16",
        weights: str = '',
        repo_dir: str = '',
) -> nn.Module:
    """
    Load DINOv3 model from torch.hub using local repository.

    DINOv3 weights require access approval from Meta AI.

    Args:
        model_name: Model variant (e.g., "dinov3_vitl16", "dinov3_vitb16")
        weights: URL or local path to checkpoint weights (obtained from Meta AI)
        repo_dir: Path to local DINOv3 repository clone

    Returns:
        DINOv3 model
    """
    model = torch.hub.load(repo_dir, model_name, source="local", weights=weights)
    model.eval()
    return model


# =======================
# Task-Specific Heads
# =======================

class ClassificationHead(nn.Module):
    """
    Linear probing classification head.

    Following DINOv3 paper: linear layer on concatenated CLS token and mean-pooled patch tokens.
    Input dimension is 2 * embed_dim due to concatenation.
    """

    def __init__(
            self,
            in_features: int,
            num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        out_features = 1 if num_classes == 2 else num_classes
        # Input is 2 * in_features due to concatenation of CLS + mean(patches)
        self.fc = nn.Linear(2 * in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Concatenated features [B, 2*D] (CLS token + mean-pooled patches)
        Returns:
            logits [B, 1] for binary classification or [B, num_classes] for multi-class classification
        """
        return self.fc(x)


class SegmentationHead(nn.Module):
    """
    Multi-scale linear segmentation head.

    Following DINOv3 implementation:
    - Multi-scale feature fusion via concatenation
    - Dropout for regularization
    - BatchNorm
    - Single 1x1 conv for classification
    - Bilinear upsampling to original image size
    """

    def __init__(
            self,
            in_channels: List[int],
            num_classes: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)  # Total channels after concatenation
        self.num_classes = num_classes

        self.bn = nn.BatchNorm2d(self.channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(self.channels, num_classes, kernel_size=1, padding=0, stride=1)

        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, features: List[Tensor], output_size: Tuple[int, int]) -> Tensor:
        """
        Forward pass with dropout (training mode).

        Args:
            features: List of [B, C_i, H_patches, W_patches] feature maps from the backbone
            output_size: Target (H, W)
        Returns:
            [B, num_classes, H, W] logits
        """
        # Concatenate along channel dimension
        x = torch.cat(features, dim=1)

        # Apply dropout, BN, and conv
        x = self.dropout(x)
        x = self.bn(x)
        logits = self.conv(x)

        # Upsample to target size
        logits = F.interpolate(
            logits, size=output_size, mode='bilinear', align_corners=False
        )

        return logits


# =====================================
# Full Models with Backbone + Head
# =====================================

class DINOClassifier(nn.Module):
    """
    DINO backbone with linear probing classification head.
    """

    def __init__(
            self,
            backbone: nn.Module,
            num_classes: int,
            feature_dim: int,
            freeze_backbone: bool = True,
            use_lora: bool = False,
            lora_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
        if use_lora and lora_config:
            self.backbone = apply_lora_to_model(
                self.backbone,
                r=lora_config.get("r", 16),
                alpha=lora_config.get("alpha", 32),
                dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"])
            )

        # Linear probing head
        self.head = ClassificationHead(feature_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]
        Returns:
            logits: [B, 1] for binary, [B, num_classes] for multi-class
        """
        # Get detailed features from backbone
        features_dict = self.backbone.forward_features(x)

        # Extract CLS token and patch tokens
        cls_token = features_dict["x_norm_clstoken"]  # [B, D]
        patch_tokens = features_dict["x_norm_patchtokens"]  # [B, N, D]

        # Concatenate CLS token with mean-pooled patch tokens
        # Following DINOv3 paper implementation
        linear_input = torch.cat(
            [
                cls_token,
                patch_tokens.mean(dim=1),  # Mean pool over spatial dimension
            ],
            dim=1,
        )  # [B, 2*D]

        # Apply classification head
        return self.head(linear_input)


class DINOSegmenter(nn.Module):
    """
    DINO backbone with multi-scale segmentation head.

    Extracts features from 4 evenly-spaced intermediate layers
    following the DINOv3 segmentation approach.
    """

    def __init__(
            self,
            backbone: nn.Module,
            num_classes: int,
            feature_dim: int,
            num_layers: int = 24,
            freeze_backbone: bool = True,
            use_lora: bool = False,
            lora_config: Optional[Dict] = None,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.num_layers = num_layers

        # Get layer indices for multi-scale feature extraction
        # Following DINOv3: for ViT-L (24 layers) we use [5, 11, 17, 23]
        self.layer_indices = self._get_layer_indices(num_layers)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Apply LoRA if requested
        if use_lora and lora_config:
            self.backbone = apply_lora_to_model(
                self.backbone,
                r=lora_config.get("r", 16),
                alpha=lora_config.get("alpha", 32),
                dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"]),
            )

        # Multi-scale head expects list of feature dimensions
        # All layers have same dim, so we replicate it
        in_channels = [feature_dim] * len(self.layer_indices)
        self.head = SegmentationHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _get_layer_indices(self, num_layers: int) -> List[int]:
        """
        Get four evenly spaced indices for multi-scale feature extraction.
        """

        return [i * (num_layers // 4) - 1 for i in range(1, 5)]

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        # Get multi-scale dense features
        features = self._get_dense_features(x)

        # Apply segmentation head
        return self.head(features, output_size=(H, W))

    def _get_dense_features(self, x: Tensor) -> List[Tensor]:
        """
        Extract dense patch features from multiple layers.

        Returns:
            List of [B, C, H_patches, W_patches] feature maps
        """
        # Use get_intermediate_layers for multi-scale dense features
        features = self.backbone.get_intermediate_layers(
            x, n=self.layer_indices, reshape=True, norm=True
        )
        return features


# =================
# Model Factory
# =================

def create_model(
        model_type: str,  # "dinov2" or "dinov3"
        task: Optional[str] = None,  # "classification" or "segmentation"
        num_classes: Optional[int] = None,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        device: str = "cuda",
        weights: str = '',
        repo_dir: str = '',
) -> nn.Module:
    """
    Factory function to create models

    Args:
        model_type: "dinov2" or "dinov3"
        task: Task type
        num_classes: Number of classes
        freeze_backbone: Whether to freeze backbone parameters
        use_lora: Whether to use LoRA fine-tuning
        lora_config: LoRA configuration dict
        device: Device to place model on
        weights: Weights file
        repo_dir: Local DINOv3 repo path

    Returns:
        Configured model
    """
    # Load backbone
    if model_type == "dinov2":
        backbone = load_dinov2("dinov2_vitl14")
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone.to(device)
    elif model_type == "dinov3":
        backbone = load_dinov3("dinov3_vitl16",
                               weights=weights,
                               repo_dir=repo_dir)
        feature_dim = 1024
        num_layers = 24
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create task-specific model
    if task == "classification":
        model = DINOClassifier(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    elif task == "segmentation":
        model = DINOSegmenter(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            num_layers=num_layers,
            freeze_backbone=freeze_backbone,
            use_lora=use_lora,
            lora_config=lora_config,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return model.to(device)


def count_parameters(model: nn.Module) -> Dict[str, int | float]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test DINOv3 with classification head
    model = create_model(
        model_type="dinov3",
        task="classification",
        num_classes=2,
        freeze_backbone=True,
        use_lora=False,
        device="mps"
    )
    print(f"DINOv3 Classifier: {model}")
    print(f"Number of parameters: {count_parameters(model)}")
    print("\n" + "=" * 60)

    # Test with LoRA
    lora_config = {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["qkv", "proj", "fc1", "fc2"]}
    model_lora = create_model(
        model_type="dinov3",
        task="classification",
        num_classes=2,
        freeze_backbone=True,
        use_lora=True,
        lora_config=lora_config,
        device="mps"
    )
    print(f"DINOv3 Classifier + LoRA: {model_lora}")
    print(f"Number of parameters: {count_parameters(model_lora)}")
    print("\n" + "=" * 60)

    # Test DINOv3 with segmentation head
    model_seg = create_model(
        model_type="dinov3",
        task="segmentation",
        num_classes=3,
        freeze_backbone=True,
        use_lora=False,
        device="mps"
    )
    print(f"DINOv3 Segmenter: {model_seg}")
    print(f"Number of parameters: {count_parameters(model_seg)}")
    print("\n" + "=" * 60)

    # Test DINOv2 model loading
    model_dinov2 = create_model(
        model_type="dinov2",
        freeze_backbone=True,
        device="mps"
    )
    print(f"DINOv2 Model: {model_dinov2}")
    print(f"Number of parameters: {count_parameters(model_dinov2)}")
