"""
Dataset Downloading and Loading Utilities
==========================================
Handles DeepGlobe, PatchCamelyon, and Coralscapes datasets
"""

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import h5py
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import dataset-specific augmentations
from augmentations import (
    get_patchcamelyon_transforms,
    get_deepglobe_transforms,
    get_coralscapes_transforms,
    get_deepglobe_joint_transforms,
    get_coralscapes_joint_transforms,
)


# =======================
# Download Utilities
# =======================

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = None):
    """Download a file from a URL"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"File already exists: {output_path}")
        return

    desc = desc or f"Downloading {output_path.name}"
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


# =======================
# PatchCamelyon Dataset
# =======================

PCAM_URLS = {
    'train_x': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz',
    'train_y': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz',
    'valid_x': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz',
    'valid_y': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz',
    'test_x': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz',
    'test_y': 'https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz',
}


def download_patchcamelyon(data_root: Path):
    """Download PatchCamelyon dataset"""
    pcam_dir = Path(data_root) / "patchcamelyon"
    pcam_dir.mkdir(parents=True, exist_ok=True)

    for name, url in PCAM_URLS.items():
        gz_path = pcam_dir / f"{name}.h5.gz"
        h5_path = pcam_dir / f"{name}.h5"

        if h5_path.exists():
            print(f"Already exists: {h5_path}")
            continue

        print(f"Downloading {name}...")
        download_file(url, gz_path, desc=f"Downloading {name}")

        print(f"Extracting {name}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(h5_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()

    print("PatchCamelyon download complete.")


class PatchCamelyonDataset(Dataset):
    """PatchCamelyon dataset for histopathology classification"""

    def __init__(
            self,
            data_root: Path,
            split: str = "train",
            transform: Optional[T.Compose] = None,
            image_size: int = 96,
    ):
        self.data_root = Path(data_root) / "patchcamelyon"
        self.split = split
        self.transform = transform
        self.image_size = image_size

        # Map split names
        split_map = {"train": "train", "val": "valid", "test": "test"}
        h5_split = split_map.get(split, split)

        # Load data
        x_path = self.data_root / f"{h5_split}_x.h5"
        y_path = self.data_root / f"{h5_split}_y.h5"

        with h5py.File(x_path, 'r') as f:
            self.images = f['x'][:]
        with h5py.File(y_path, 'r') as f:
            self.labels = f['y'][:].flatten()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])

        # Resize
        image = TF.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return image, label


# ===============================
# DeepGlobe Land Cover Dataset
# ===============================

# DEEPGLOBE_CLASSES = {
#     0: "urban_land",
#     1: "agriculture_land",
#     2: "rangeland",
#     3: "forest_land",
#     4: "water",
#     5: "barren_land",
#     6: "unknown"
# }

# RGB values for each class
DEEPGLOBE_COLORS = {
    (0, 255, 255): 0,  # urban_land - cyan
    (255, 255, 0): 1,  # agriculture_land - yellow
    (255, 0, 255): 2,  # rangeland - magenta
    (0, 255, 0): 3,  # forest_land - green
    (0, 0, 255): 4,  # water - blue
    (255, 255, 255): 5,  # barren_land - white
    (0, 0, 0): 6,  # unknown - black
}


def download_deepglobe(data_root: Path):
    """
    Download DeepGlobe Land Cover dataset
    """
    deepglobe_dir = Path(data_root) / "deepglobe"
    deepglobe_dir.mkdir(parents=True, exist_ok=True)

    train_dir = deepglobe_dir / "train"
    if train_dir.exists() and any(train_dir.glob("*_sat.jpg")):
        print("DeepGlobe dataset found.")
        return True

    # Try Kaggle download if kaggle is installed
    try:
        import kaggle
        print("Attempting Kaggle download...")
        kaggle.api.dataset_download_files(
            'balraj98/deepglobe-land-cover-classification-dataset',
            path=deepglobe_dir,
            unzip=True
        )
        print("DeepGlobe download complete.")
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")

        # Create placeholder directory structure
        train_dir.mkdir(parents=True, exist_ok=True)
        return False


def rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB mask to class indices"""
    mask_class = np.zeros(mask_rgb.shape[:2], dtype=np.int64)

    for rgb, class_idx in DEEPGLOBE_COLORS.items():
        matches = np.all(mask_rgb == rgb, axis=-1)
        mask_class[matches] = class_idx

    return mask_class


class DeepGlobeDataset(Dataset):
    """DeepGlobe Land Cover Classification dataset"""

    def __init__(
            self,
            data_root: Path,
            split: str = "train",
            transform: Optional[T.Compose] = None,
            joint_transform: Optional[Any] = None,  # For geometric augmentations on both image and mask
            image_size: int = 512,
    ):
        self.data_root = Path(data_root) / "deepglobe"
        self.split = split
        self.transform = transform
        self.joint_transform = joint_transform
        self.image_size = image_size

        # Find all images
        train_dir = self.data_root / "train"
        all_images = sorted(train_dir.glob("*_sat.jpg"))

        # Create train/val/test splits (803 images total)
        np.random.seed(42)
        indices = np.random.permutation(len(all_images))

        n_train = 602
        n_val = 80

        if split == "train":
            self.image_paths = [all_images[i] for i in indices[:n_train]]
        elif split == "val":
            self.image_paths = [all_images[i] for i in indices[n_train:n_train + n_val]]
        else:  # test
            self.image_paths = [all_images[i] for i in indices[n_train + n_val:]]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = str(img_path).replace("_sat.jpg", "_mask.png")

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # Convert RGB mask to class indices
        mask = rgb_to_class(mask_rgb)
        mask = Image.fromarray(mask.astype(np.uint8))

        # Resize
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size),
                         interpolation=T.InterpolationMode.NEAREST)

        # Apply joint geometric augmentations (same to image and mask)
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        # Apply image-only transforms (color augmentation, normalization)
        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


# ============================================================
# Coralscapes Dataset (Underwater Coral Reef Segmentation)
# ============================================================

# Coralscapes actually has 40 classes (labels 0-39 inclusive)
CORALSCAPES_NUM_CLASSES = 40


# CORALSCAPES_CLASSES = []


def download_coralscapes(data_root: Path):
    """
    Download Coralscapes dataset from Hugging Face
    """
    coralscapes_dir = Path(data_root) / "coralscapes"
    coralscapes_dir.mkdir(parents=True, exist_ok=True)

    cache_path = coralscapes_dir / "dataset"
    if cache_path.exists():
        print("Coralscapes dataset found.")
        return True

    try:
        from datasets import load_dataset

        print("Downloading Coralscapes dataset from Hugging Face...")
        dataset = load_dataset("EPFL-ECEO/coralscapes", cache_dir=str(coralscapes_dir), token=True)

        dataset.save_to_disk(str(cache_path))

        print("Coralscapes download complete.")
        return True

    except ImportError:
        print("Please install the 'datasets' library: pip install datasets")
        return False
    except Exception as e:
        print(f"Error downloading Coralscapes: {e}")
        return False


class CoralscapesDataset(Dataset):
    """
    Coralscapes coral reef semantic segmentation dataset
    
    The first general-purpose dense semantic segmentation dataset for coral reefs.
    - 2075 images at 2048x1024px resolution
    - 39 benthic classes
    - High domain shift (underwater coral reef imagery)
    
    Reference: Sauder et al., "The Coralscapes Dataset: Semantic Scene 
    Understanding in Coral Reefs"
    """

    def __init__(
            self,
            data_root: Path,
            split: str = "train",
            transform: Optional[T.Compose] = None,
            joint_transform: Optional[Any] = None,
            image_size: Tuple[int, int] = (512, 1024),
    ):
        self.data_root = Path(data_root) / "coralscapes"
        self.split = split
        self.transform = transform
        self.joint_transform = joint_transform
        self.image_size = image_size
        self.num_classes = CORALSCAPES_NUM_CLASSES

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from Hugging Face cache or download"""
        try:
            from datasets import load_from_disk, load_dataset

            # Check if cached version exists
            cache_path = self.data_root / "dataset"
            if cache_path.exists():
                full_dataset = load_from_disk(str(cache_path))
            else:
                print("Downloading Coralscapes dataset...")
                full_dataset = load_dataset(
                    "EPFL-ECEO/coralscapes",
                    cache_dir=str(self.data_root)
                )
                full_dataset.save_to_disk(str(cache_path))

            # Map split names (Coralscapes uses train/validation/test)
            split_map = {"train": "train", "val": "validation", "test": "test"}
            hf_split = split_map.get(self.split, self.split)

            if hf_split in full_dataset:
                self.dataset = full_dataset[hf_split]
                print(f"Loaded Coralscapes {self.split} split: {len(self.dataset)} images")
            else:
                raise ValueError(f"Split '{self.split}' not found in Coralscapes dataset")

        except Exception as e:
            raise RuntimeError(f"Error loading Coralscapes: {e}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image and label from HF dataset
        image = item["image"]
        mask = item["label"]

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        # Resize maintaining aspect ratio (original is 2048x1024, i.e., 1:2)
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size,
                         interpolation=T.InterpolationMode.NEAREST)

        # Apply joint geometric augmentations (same to image and mask)
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        # Apply image-only transforms (underwater color augmentation, normalization)
        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


# =====================
# DataLoader Factory
# =====================

def create_dataloaders(
        dataset_name: str,
        data_root: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int | Tuple[int, int] | None = None,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for a given dataset with domain-specific augmentations.
    """

    data_root = Path(data_root)

    if dataset_name == "patchcamelyon":
        # We use histopathology-specific augmentations
        train_transform = get_patchcamelyon_transforms(image_size, is_train=True)
        eval_transform = get_patchcamelyon_transforms(image_size, is_train=False)

        train_dataset = PatchCamelyonDataset(
            data_root, split="train", transform=train_transform,
        )
        val_dataset = PatchCamelyonDataset(
            data_root, split="val", transform=eval_transform,
        )
        test_dataset = PatchCamelyonDataset(
            data_root, split="test", transform=eval_transform,
        )

    elif dataset_name == "deepglobe":
        # We use satellite-specific augmentations
        # Joint transform for geometric augmentations (applied to both image and mask)
        train_joint_transform = get_deepglobe_joint_transforms(image_size)
        # Image-only transform for color augmentations and normalization
        train_transform = get_deepglobe_transforms(image_size, is_train=True)
        eval_transform = get_deepglobe_transforms(image_size, is_train=False)

        train_dataset = DeepGlobeDataset(
            data_root, split="train",
            transform=train_transform,
            joint_transform=train_joint_transform,
            image_size=image_size
        )
        val_dataset = DeepGlobeDataset(
            data_root, split="val",
            transform=eval_transform,
            joint_transform=None,  # No augmentation for validation
            image_size=image_size
        )
        test_dataset = DeepGlobeDataset(
            data_root, split="test",
            transform=eval_transform,
            joint_transform=None,
            image_size=image_size
        )

    elif dataset_name == "coralscapes":
        # We use underwater-specific augmentations
        # Joint transform for geometric augmentations
        train_joint_transform = get_coralscapes_joint_transforms(image_size)
        # Image-only transform for underwater color augmentations and normalization
        train_transform = get_coralscapes_transforms(image_size, is_train=True)
        eval_transform = get_coralscapes_transforms(image_size, is_train=False)

        train_dataset = CoralscapesDataset(
            data_root, split="train",
            transform=train_transform,
            joint_transform=train_joint_transform,
            image_size=image_size
        )
        val_dataset = CoralscapesDataset(
            data_root, split="val",
            transform=eval_transform,
            joint_transform=None,  # No augmentation for validation
            image_size=image_size
        )
        test_dataset = CoralscapesDataset(
            data_root, split="test",
            transform=eval_transform,
            joint_transform=None,
            image_size=image_size
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


# ========================
# Main Download Function
# ========================

def download_all_datasets(data_root: Path):
    """Download all required datasets"""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading PatchCamelyon...")
    print("=" * 60)
    download_patchcamelyon(data_root)

    print("\n" + "=" * 60)
    print("Downloading DeepGlobe...")
    print("=" * 60)
    download_deepglobe(data_root)

    print("\n" + "=" * 60)
    print("Downloading Coralscapes...")
    print("=" * 60)
    download_coralscapes(data_root)

    print("\n" + "=" * 60)
    print("Dataset setup complete.")
    print("=" * 60)


if __name__ == "__main__":
    download_all_datasets(Path("/Users/gloirelinvani/Downloads/data"))
