"""
Training Module
===============
Training loops, loss functions, and evaluation metrics
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score
)
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============
# Metrics
# ============

@dataclass
class ClassificationMetrics:
    """Metrics for classification tasks"""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation tasks"""
    mean_iou: float
    class_iou: Dict[int, float]
    mean_dice: float
    class_dice: Dict[int, float]
    pixel_accuracy: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_classification_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
) -> ClassificationMetrics:
    """Compute classification metrics"""
    return ClassificationMetrics(
        accuracy=accuracy_score(targets, predictions),
        f1_macro=f1_score(targets, predictions, average='macro', zero_division=0),
        f1_weighted=f1_score(targets, predictions, average='weighted', zero_division=0),
        precision_macro=precision_score(targets, predictions, average='macro', zero_division=0),
        recall_macro=recall_score(targets, predictions, average='macro', zero_division=0),
    )


def compute_segmentation_metrics(
        predictions: np.ndarray,  # [B, H, W]
        targets: np.ndarray,  # [B, H, W]
        num_classes: int,
        ignore_index: int = 255,
) -> SegmentationMetrics:
    """Compute segmentation metrics"""
    # Flatten
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Pixel accuracy
    pixel_accuracy = (pred_flat == target_flat).mean()

    # Per-class IoU and Dice
    class_iou = {}
    class_dice = {}

    for cls in range(num_classes):
        pred_cls = pred_flat == cls
        target_cls = target_flat == cls

        target_sum = target_cls.sum()

        # Only compute metrics for classes present in ground truth
        if target_sum > 0:
            pred_sum = pred_cls.sum()
            intersection = (pred_cls & target_cls).sum()
            union = (pred_cls | target_cls).sum()

            # IoU
            iou = intersection / union
            class_iou[cls] = float(iou)

            # Dice
            dice = 2 * intersection / (pred_sum + target_sum)
            class_dice[cls] = float(dice)

    mean_iou = np.mean(list(class_iou.values())) if class_iou else 0.0
    mean_dice = np.mean(list(class_dice.values())) if class_dice else 0.0

    return SegmentationMetrics(
        pixel_accuracy=float(pixel_accuracy),
        mean_iou=float(mean_iou),
        class_iou=class_iou,
        mean_dice=float(mean_dice),
        class_dice=class_dice,
    )


# ===================
# Loss Functions
# ===================

class JaccardLoss(nn.Module):
    """Jaccard Loss (IoU Loss) for segmentation"""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        num_classes = inputs.shape[1]

        # Softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        targets_onehot = F.one_hot(
            targets, num_classes
        ).permute(0, 3, 1, 2).float()

        # Compute Jaccard (IoU)
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        sum_probs_targets = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        union = sum_probs_targets - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_per_class = iou.mean(dim=0)  # mean over batch

        return 1 - iou_per_class.mean()  # mean over classes


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        num_classes = inputs.shape[1]

        # Softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        targets_onehot = F.one_hot(
            targets, num_classes  # clamp(0, num_classes - 1)
        ).permute(0, 3, 1, 2).float()

        # Compute Dice
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        sum_probs_targets = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (sum_probs_targets + self.smooth)
        dice_per_class = dice.mean(dim=0)  # mean over batch

        return 1 - dice_per_class.mean()  # mean over classes


class SegmentationLoss(nn.Module):
    """Combined loss for segmentation"""

    def __init__(
            self,
            ce_weight: float = 1 / 2,
            jaccard_weight: float = 1 / 2,
            dice_weight: float = 0.0,
            ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.jaccard_weight = jaccard_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=1e-5, ignore_index=ignore_index)
        self.jaccard_loss = JaccardLoss(smooth=1e-5, ignore_index=ignore_index)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        metric_based_loss = 0.0
        if self.dice_weight > 0.0:
            metric_based_loss += self.dice_weight * self.dice_loss(inputs, targets)
        if self.jaccard_weight > 0.0:
            metric_based_loss += self.jaccard_weight * self.jaccard_loss(inputs, targets)
        return metric_based_loss + self.ce_weight * self.ce_loss(inputs, targets)


# ======================
# Training Loop
# ======================

class Trainer:
    """Unified trainer for all tasks"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None,
            task: str = "classification",  # "classification", "segmentation", "multilabel"
            ce_weight: float = 1 / 2,
            jaccard_weight: float = 1 / 2,
            dice_weight: float = 0.0,
            ignore_index: int = 255,
            num_classes: int = 2,
            learning_rate: float = 1e-3,
            weight_decay: float = 0.01,
            epochs: int = 50,
            warmup_ratio: float = 0.1,  # fraction of total steps for warmup
            device: str = "cuda",
            output_dir: Path = Path("./outputs"),
            use_lora: bool = False,
            lora_lr: Optional[float] = None,
            early_stopping_patience: int = 10,
            early_stopping_min_delta: float = 0.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task = task
        self.num_classes = num_classes
        self.epochs = epochs
        self.max_epochs = epochs  # Store for scheduler calculation
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_lora = use_lora
        self.learning_rate = learning_rate

        # Setup loss function
        if task == "classification":
            if num_classes == 2:
                # Binary classification: BCEWithLogitsLoss for single logit output
                self.criterion = nn.BCEWithLogitsLoss()
                self.is_binary_classification = True
            else:
                # Multi-class classification: CrossEntropyLoss
                self.criterion = nn.CrossEntropyLoss()
                self.is_binary_classification = False
        elif task == "segmentation":
            self.criterion = SegmentationLoss(ce_weight=ce_weight,
                                              jaccard_weight=jaccard_weight,
                                              dice_weight=dice_weight,
                                              ignore_index=ignore_index)
            self.is_binary_classification = False
        else:
            raise ValueError(f"Unknown task: {task}")

        # Setup optimizer
        if use_lora:
            # Separate parameter groups for LoRA and head
            lora_params = []
            head_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'lora' in name.lower():
                        lora_params.append(param)
                    else:
                        head_params.append(param)

            param_groups = [
                {'params': head_params, 'lr': learning_rate},
                {'params': lora_params, 'lr': lora_lr or learning_rate * 0.1},
            ]
            self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        else:
            # Only train head parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        # Setup step-based scheduler
        steps_per_epoch = len(train_loader)
        total_steps = self.max_epochs * steps_per_epoch
        warmup_steps = int(warmup_ratio * total_steps)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=learning_rate * 0.01
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )

        print(f"Scheduler setup: {steps_per_epoch} steps/epoch, {total_steps} total steps, {warmup_steps} warmup steps")

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epochs_without_improvement = 0

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'test_metrics': None,
        }
        self.best_metric = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Handle binary classification target shape
            if self.task == "classification" and self.is_binary_classification:
                # BCEWithLogitsLoss expects targets of the same shape as outputs
                # outputs: [B, 1], targets: [B] -> targets: [B, 1]
                targets = targets.float().unsqueeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()  # Step-based scheduler update

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "Eval") -> Tuple[float, Any]:
        """Evaluate on a dataset"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []  # Store probabilities for AUC computation

        pbar = tqdm(loader, desc=desc)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            # Compute loss
            if self.task == "classification" and self.is_binary_classification:
                loss = self.criterion(outputs, targets.float().unsqueeze(1))
            else:
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            if self.task == "classification":
                if self.is_binary_classification:
                    # Binary: sigmoid then threshold at 0.5
                    probs = torch.sigmoid(outputs).squeeze(1)  # [B]
                    preds = (probs > 0.5).long().cpu().numpy()
                    all_probabilities.append(probs.cpu().numpy())
                else:
                    # Multi-class: argmax
                    preds = outputs.argmax(dim=1).cpu().numpy()
            elif self.task == "segmentation":
                preds = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.append(preds)
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / num_batches

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        if self.task == "classification":
            metrics = compute_classification_metrics(all_predictions, all_targets)
            # Add AUC for binary classification
            if self.is_binary_classification and len(all_probabilities) > 0:
                all_probabilities = np.concatenate(all_probabilities, axis=0)
                try:
                    auc = roc_auc_score(all_targets, all_probabilities)
                    print(f"  Binary AUC: {auc:.4f}")
                except Exception as e:
                    print(f"  Could not compute AUC: {e}")
        elif self.task == "segmentation":
            metrics = compute_segmentation_metrics(
                all_predictions, all_targets, self.num_classes
            )

        return avg_loss, metrics

    def get_primary_metric(self, metrics: Any) -> float:
        """Get the primary metric for model selection"""
        if self.task == "classification":
            return metrics.f1_macro
        elif self.task == "segmentation":
            return metrics.mean_iou

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Load model checkpoint and restore training state

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            The epoch number to resume from (saved_epoch + 1)
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        self.history = checkpoint['history']
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint.get('best_epoch', None)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', None)

        # If not available, try to load from best checkpoint
        if self.best_epoch is None or self.epochs_without_improvement is None:
            # Try to construct best checkpoint path
            best_checkpoint_path = checkpoint_path.parent / 'checkpoint_best.pt'

            if best_checkpoint_path.exists():
                print(f"Loading best checkpoint to retrieve missing metadata from: {best_checkpoint_path}")
                best_checkpoint = torch.load(best_checkpoint_path, map_location=self.device)

                # Get best_epoch from best checkpoint
                self.best_epoch = best_checkpoint['epoch']

                # Infer epochs_without_improvement
                current_epoch = checkpoint['epoch']
                self.epochs_without_improvement = current_epoch - self.best_epoch
            else:
                # Fallback to defaults
                print("Best checkpoint not found, defaulting best_epoch and epochs_without_improvement to 0")
                self.best_epoch = 0
                self.epochs_without_improvement = 0

        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best metric so far: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")

        return start_epoch

    def train(self) -> Dict:
        """Full training loop"""
        print(f"\n{'=' * 60}")
        print(f"Training {self.task} model")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of training batches: {len(self.train_loader)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of Validation batches: {len(self.val_loader)}")
        if self.test_loader:
            print(f"Number of test samples: {len(self.test_loader.dataset)}")
            print(f"Number of Test batches: {len(self.test_loader)}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(self.val_loader, "Val")

            # Get primary metric (scheduler is updated per-step in train_epoch)
            primary_metric = self.get_primary_metric(val_metrics)

            # Check if best (with min_delta)
            is_best = primary_metric >= self.best_metric + self.early_stopping_min_delta
            if is_best:
                self.best_metric = primary_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics.to_dict())

            # Print progress
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metric: {primary_metric:.4f}")
            if self.task == "classification":
                print(f"  Val Accuracy: {val_metrics.accuracy:.4f}")
            if is_best:
                print(f"  [NEW BEST]")

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'=' * 60}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {self.early_stopping_patience} consecutive epochs")
                print(f"Best metric: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
                print(f"{'=' * 60}")
                break

        # Final evaluation on test set
        if self.test_loader:
            print(f"\n{'=' * 60}")
            print("Final Test Evaluation")
            print(f"{'=' * 60}")

            # Load best model
            best_checkpoint = torch.load(self.output_dir / 'checkpoint_best.pt')
            self.model.load_state_dict(best_checkpoint['model_state_dict'])

            test_loss, test_metrics = self.evaluate(self.test_loader, "Test")
            self.history['test_metrics'] = test_metrics.to_dict()

            print(f"Test Loss: {test_loss:.4f}")
            metrics_dict = test_metrics.to_dict()
            # Format metrics including nested dictionaries
            metrics_parts = []
            for k, v in metrics_dict.items():
                if isinstance(v, dict):
                    # Format nested dictionary
                    nested_str = "{" + ", ".join([f"{nk}: {nv:.2f}" for nk, nv in v.items()]) + "}"
                    metrics_parts.append(f"{k}: {nested_str}")
                elif isinstance(v, (int, float)):
                    metrics_parts.append(f"{k}: {v:.2f}")
            metrics_str = ", ".join(metrics_parts)
            print(f"Test Metrics: {metrics_str}")

        # Save final history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def resume_training(self, checkpoint_path: Path, total_epochs: int) -> Dict:
        """
        Resume training from a checkpoint

        Args:
            checkpoint_path: Path to checkpoint file to resume from
            total_epochs: Total number of epochs to train for (including already completed epochs)

        Returns:
            Training history dictionary
        """
        # Load checkpoint and get starting epoch
        start_epoch = self.load_checkpoint(checkpoint_path)

        # Update epochs to reflect the total target
        remaining_epochs = total_epochs - start_epoch
        if remaining_epochs <= 0:
            print(f"\n{'=' * 60}")
            print(f"Warning: Total epochs ({total_epochs}) <= completed epochs ({start_epoch})")
            print(f"No additional training needed.")
            print(f"{'=' * 60}\n")
            return self.history

        print(f"\n{'=' * 60}")
        print(f"Resuming training for {self.task} model")
        print(f"{'=' * 60}")
        print(f"Device: {self.device}")
        print(f"Starting epoch: {start_epoch}")
        print(f"Target total epochs: {total_epochs}")
        print(f"Remaining epochs: {remaining_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        if self.test_loader:
            print(f"Number of test samples: {len(self.test_loader.dataset)}")
        print(f"{'=' * 60}\n")

        # Update self.epochs to reflect total target
        self.epochs = total_epochs

        # Training loop (same as train method)
        for epoch in range(start_epoch, total_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(self.val_loader, "Val")

            # Get primary metric
            primary_metric = self.get_primary_metric(val_metrics)

            # Check if best
            is_best = primary_metric >= self.best_metric + self.early_stopping_min_delta
            if is_best:
                self.best_metric = primary_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics.to_dict())

            # Print progress
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metric: {primary_metric:.4f}")
            if self.task == "classification":
                print(f"  Val Accuracy: {val_metrics.accuracy:.4f}")
            if is_best:
                print(f"  [NEW BEST]")

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'=' * 60}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {self.early_stopping_patience} consecutive epochs")
                print(f"Best metric: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
                print(f"{'=' * 60}")
                break

        # Final evaluation on test set
        if self.test_loader:
            print(f"\n{'=' * 60}")
            print("Final Test Evaluation")
            print(f"{'=' * 60}")

            # Load best model
            best_checkpoint = torch.load(self.output_dir / 'checkpoint_best.pt')
            self.model.load_state_dict(best_checkpoint['model_state_dict'])

            test_loss, test_metrics = self.evaluate(self.test_loader, "Test")
            self.history['test_metrics'] = test_metrics.to_dict()

            print(f"Test Loss: {test_loss:.4f}")
            metrics_dict = test_metrics.to_dict()
            # Format metrics including nested dictionaries
            metrics_parts = []
            for k, v in metrics_dict.items():
                if isinstance(v, dict):
                    # Format nested dictionary
                    nested_str = "{" + ", ".join([f"{nk}: {nv:.2f}" for nk, nv in v.items()]) + "}"
                    metrics_parts.append(f"{k}: {nested_str}")
                elif isinstance(v, (int, float)):
                    metrics_parts.append(f"{k}: {v:.2f}")
            metrics_str = ", ".join(metrics_parts)
            print(f"Test Metrics: {metrics_str}")

        # Save final history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history


if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics computation...")

    # Classification
    preds = np.array([0, 1, 1, 0, 1])
    targets = np.array([0, 1, 0, 0, 1])
    metrics = compute_classification_metrics(preds, targets)
    print(f"Classification metrics: {metrics.to_dict()}")

    # Segmentation
    preds = np.random.randint(0, 7, (2, 32, 32))
    targets = np.random.randint(0, 7, (2, 32, 32))
    metrics = compute_segmentation_metrics(preds, targets, num_classes=7)
    print(f"Segmentation metrics: {metrics.to_dict()}")
