"""DINOv2 and DINOv3 backbones with task-specific heads and LoRA fine-tuning."""

import math
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRALayer(nn.Module):
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

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        return self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)


class LinearWithLoRA(nn.Module):
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
    for name, module in model.named_modules():
        should_apply = any(target in name for target in target_modules)

        if should_apply and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            lora_linear = LinearWithLoRA(
                module, r=r, alpha=alpha, dropout=dropout
            )
            setattr(parent, attr_name, lora_linear)

    return model


def load_dinov2(model_name: str = "dinov2_vitl14", pretrained: bool = True) -> nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
    model.eval()
    return model


def load_dinov3(
        model_name: str = "dinov3_vitl16",
        weights: str = '',
        repo_dir: str = '',
) -> nn.Module:
    """Load DINOv3 via torch.hub from a local repository clone (weights require Meta AI's approval)."""
    model = torch.hub.load(repo_dir, model_name, source="local", weights=weights)
    model.eval()
    return model


class ClassificationHead(nn.Module):
    """Linear head on the CLS token concatenated with the mean-pooled patch tokens, as in DINOv3."""

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
        return self.fc(x)


class SegmentationHead(nn.Module):
    """Multi-scale linear segmentation head, following the DINOv3 implementation."""

    def __init__(
            self,
            in_channels: List[int],
            num_classes: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        self.num_classes = num_classes

        self.bn = nn.BatchNorm2d(self.channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(self.channels, num_classes, kernel_size=1, padding=0, stride=1)

        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, features: List[Tensor], output_size: Tuple[int, int]) -> Tensor:
        x = torch.cat(features, dim=1)

        x = self.dropout(x)
        x = self.bn(x)
        logits = self.conv(x)

        logits = F.interpolate(
            logits, size=output_size, mode='bilinear', align_corners=False
        )

        return logits


class DINOClassifier(nn.Module):
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

        if use_lora and lora_config:
            self.backbone = apply_lora_to_model(
                self.backbone,
                r=lora_config.get("r", 16),
                alpha=lora_config.get("alpha", 32),
                dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"])
            )

        self.head = ClassificationHead(feature_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features_dict = self.backbone.forward_features(x)

        cls_token = features_dict["x_norm_clstoken"]  # [B, D]
        patch_tokens = features_dict["x_norm_patchtokens"]  # [B, N, D]

        # Following DINOv3 paper implementation
        linear_input = torch.cat(
            [
                cls_token,
                patch_tokens.mean(dim=1),
            ],
            dim=1,
        )  # [B, 2*D]

        return self.head(linear_input)


class DINOSegmenter(nn.Module):
    """DINO backbone with a segmentation head on four evenly spaced intermediate layers, as in DINOv3."""

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

        # Following DINOv3: for ViT-L (24 layers) we use [5, 11, 17, 23]
        self.layer_indices = self._get_layer_indices(num_layers)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_lora and lora_config:
            self.backbone = apply_lora_to_model(
                self.backbone,
                r=lora_config.get("r", 16),
                alpha=lora_config.get("alpha", 32),
                dropout=lora_config.get("dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"]),
            )

        # All layers have same dim, so we replicate it
        in_channels = [feature_dim] * len(self.layer_indices)
        self.head = SegmentationHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _get_layer_indices(self, num_layers: int) -> List[int]:
        return [i * (num_layers // 4) - 1 for i in range(1, 5)]

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        features = self._get_dense_features(x)

        return self.head(features, output_size=(H, W))

    def _get_dense_features(self, x: Tensor) -> List[Tensor]:
        """Return one [B, C, H_patches, W_patches] feature map per selected layer."""
        features = self.backbone.get_intermediate_layers(
            x, n=self.layer_indices, reshape=True, norm=True
        )
        return features


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
    """Create a DINOv2 or DINOv3 model with a classification or segmentation head, and optional LoRA."""
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
    print("Testing model creation...")

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

    model_dinov2 = create_model(
        model_type="dinov2",
        freeze_backbone=True,
        device="mps"
    )
    print(f"DINOv2 Model: {model_dinov2}")
    print(f"Number of parameters: {count_parameters(model_dinov2)}")
