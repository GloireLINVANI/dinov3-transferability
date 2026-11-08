"""
Dataset-Specific Data Augmentations
====================================
Custom augmentations tailored for each domain:
- PatchCamelyon: Histopathology stain augmentation
- DeepGlobe: Satellite imagery augmentation
- Coralscapes: Underwater image augmentation
"""

import random
from typing import Tuple

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


# ============================
# Custom Transform Classes
# ============================

class RandomRotation90:
    """Random rotation by 90 degree increments (0, 90, 180, 270)"""

    def __call__(self, img):
        angle = random.choices([0, 90, 180, 270], weights=[0.4, 0.2, 0.2, 0.2])[0]
        return TF.rotate(img, angle)


class StainAugmentation:
    """
    Simplified stain augmentation for H&E histopathology images.
    Perturbs color channels to simulate staining variation.
    
    H&E staining: Hematoxylin (blue/purple for nuclei), Eosin (pink for cytoplasm)
    """

    def __init__(
            self,
            sigma_hue: float = 0.05,
            sigma_brightness: float = 0.2,
    ):
        self.sigma_hue = sigma_hue
        self.sigma_brightness = sigma_brightness

    def __call__(self, img):
        # Convert to numpy for manipulation
        img_np = np.array(img).astype(np.float32) / 255.0

        # Perturb each channel slightly differently to simulate stain variation
        # This mimics different H&E staining protocols
        for c in range(3):
            factor = 1.0 + np.random.normal(0, self.sigma_brightness)
            img_np[:, :, c] = np.clip(img_np[:, :, c] * factor, 0, 1)

        # Convert back to PIL
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)


class HistopathologyColorJitter:
    """
    Color jitter specifically tuned for H&E stained images.
    More aggressive on saturation/hue to simulate staining variation.
    """

    def __init__(
            self,
            brightness: float = 0.25,
            contrast: float = 0.25,
            saturation: float = 0.3,
            hue: float = 0.08,
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img):
        return self.transform(img)


class GaussianBlur:
    """Gaussian blur to simulate focus variation"""

    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 1.5)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class ElasticDeformation:
    """
    Simple elastic deformation for histopathology.
    Simulates tissue deformation during slide preparation.
    """

    def __init__(self, alpha: float = 50, sigma: float = 5, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = np.array(img)
        shape = img_np.shape[:2]

        # Generate random displacement fields
        dx = np.random.randn(*shape) * self.alpha
        dy = np.random.randn(*shape) * self.alpha

        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, self.sigma)
        dy = gaussian_filter(dy, self.sigma)

        # Create meshgrid and apply displacements
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x_new = np.clip(x + dx, 0, shape[1] - 1).astype(np.int32)
        y_new = np.clip(y + dy, 0, shape[0] - 1).astype(np.int32)

        # Apply transformation
        img_deformed = img_np[y_new, x_new]

        return Image.fromarray(img_deformed)


# ===============================================
# Satellite Imagery Augmentations (DeepGlobe)
# ===============================================

class SatelliteColorJitter:
    """
    Color jitter tuned for satellite imagery.
    Simulates atmospheric conditions and seasonal variation.
    """

    def __init__(
            self,
            brightness: float = 0.2,
            contrast: float = 0.2,
            saturation: float = 0.2,
            hue: float = 0.05,  # We keep hue low, for more realistic colors
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img):
        return self.transform(img)


class RandomRotationContinuous:
    """Random rotation by any angle (for overhead imagery)"""

    def __init__(self, degrees: float = 180):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(img, angle, fill=0)


class SegmentationRandomRotation:
    """
    Apply same random rotation to both image and mask.
    For use with segmentation datasets.
    """

    def __init__(self, degrees: float = 180, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img, mask):
        if random.random() <= self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = TF.rotate(img, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        return img, mask


class SegmentationRandomFlip:
    """Apply the same random flip to both image and mask"""

    def __init__(self, horizontal: bool = True, vertical: bool = True):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, img, mask):
        if self.horizontal and random.random() <= 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if self.vertical and random.random() <= 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask


class SegmentationRandomResizedCrop:
    """Random resized crop applied to both image and mask"""

    def __init__(
            self,
            size: Tuple[int, int],
            scale: Tuple[float, float] = (0.5, 1.0),
            ratio: Tuple[float, float] = (0.8, 1.2),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, mask):
        # Get crop parameters
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

        # Apply to both
        img = TF.resized_crop(img, i, j, h, w, self.size)
        mask = TF.resized_crop(mask, i, j, h, w, self.size,
                               interpolation=T.InterpolationMode.NEAREST)
        return img, mask


# ===========================================
# Underwater Augmentations (Coralscapes)
# ===========================================

class UnderwaterBackscatter:
    """
    Simulates backscatter/haze from particles in water.
    Adds a veiling light that reduces contrast.
    """

    def __init__(
            self,
            intensity_range: Tuple[float, float] = (0.0, 0.15),
            color: Tuple[float, float, float] = (0.6, 0.8, 0.9),  # Blueish haze
    ):
        self.intensity_range = intensity_range
        self.color = np.array(color)

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0

        intensity = random.uniform(*self.intensity_range)

        # Add veiling light (backscatter)
        haze = self.color * intensity
        img_np = img_np * (1 - intensity) + haze

        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)

        return Image.fromarray(img_np)


class UnderwaterTurbidity:
    """
    Simulates turbidity (blur from particles in water).
    Underwater images are naturally softer due to light scattering.
    """

    def __init__(
            self,
            blur_range: Tuple[float, float] = (0.5, 2.0),
            p: float = 0.5,
    ):
        self.blur_range = blur_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        sigma = random.uniform(*self.blur_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class UnderwaterColorJitter:
    """
    Color jitter tuned for underwater imagery.
    Higher hue variation to account for extreme color casts.
    """

    def __init__(
            self,
            brightness: float = 0.3,
            contrast: float = 0.3,
            saturation: float = 0.3,
            hue: float = 0.12,  # Higher than the other datasets due to underwater color variation
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img):
        return self.transform(img)


class SegmentationLimitedRotation:
    """Limited rotation for segmentation images where gravity should be considered"""

    def __init__(self, degrees: float = 15, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img, mask):
        if random.random() <= self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = TF.rotate(img, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        return img, mask


# =================================
# Composite Transform Pipelines
# =================================

class PatchCamelyonAugmentation:
    """
    Complete augmentation pipeline for PatchCamelyon (histopathology).
    
    Key augmentations:
    - 90° rotation increments
    - Horizontal and vertical flips
    - Stain/color variation
    - Gaussian blur (focus variation)
    - Elastic deformation (tissue manipulation)
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

        # Geometric
        self.random_rotation_90 = RandomRotation90()

        # Color/Stain
        self.stain_aug = StainAugmentation(
            sigma_brightness=0.15,
        )
        self.color_jitter = HistopathologyColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.3,
            hue=0.08,
        )

        # Blur
        self.blur = GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))

        # Elastic deformation
        self.elastic_deform = ElasticDeformation(
            alpha=5,
            sigma=4,
            p=0.3,
        )

        # Final transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        # self.resize = T.Resize(image_size)

    def __call__(self, img):
        # img = self.resize(img)

        # Geometric augmentations
        # img = self.elastic_deform(img)
        img = self.random_rotation_90(img)
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)

        # Stain, Color augmentations
        if random.random() > 0.5:
            img = self.stain_aug(img)
        if random.random() > 0.5:
            img = self.color_jitter(img)

        # Blur
        if random.random() > 0.7:
            img = self.blur(img)

        # Convert to tensor
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img


class DeepGlobeAugmentation:
    """
    Complete augmentation pipeline for DeepGlobe (satellite imagery).
    
    Key augmentations:
    - Continuous rotation
    - Horizontal and vertical flips
    - Random resized crop (multi-scale)
    - Color jitter (atmospheric/seasonal)
    
    This only handles the image. The mask's augmentation is done in the dataset class.
    """

    def __init__(self, image_size: int = 512):
        self.image_size = image_size

        # Color
        self.color_jitter = SatelliteColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

        # Final transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def __call__(self, img):
        # Color augmentation
        if random.random() > 0.3:
            img = self.color_jitter(img)

        # Convert to tensor
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img


class CoralscapesAugmentation:
    """
    Complete augmentation pipeline for Coralscapes (underwater coral reefs).
    
    Key augmentations:
    - Limited rotation (gravity matters)
    - Horizontal flip only (no vertical - gravity)
    - Backscatter (particle haze)
    - Turbidity blur
    
    This only handles the image. The mask's augmentation is done in the dataset class.
    """

    def __init__(self, image_size: int = 512):
        self.image_size = image_size

        self.backscatter = UnderwaterBackscatter(
            intensity_range=(0.0, 0.12),
        )
        self.turbidity = UnderwaterTurbidity(
            blur_range=(0.3, 1.5),
            p=0.4,
        )
        self.color_jitter = UnderwaterColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.1,
        )

        # Final transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def __call__(self, img):
        # Underwater color augmentations
        if random.random() > 0.5:
            img = self.backscatter(img)
        if random.random() > 0.5:
            img = self.color_jitter(img)

        # Turbidity
        img = self.turbidity(img)

        # Convert to tensor
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img


# ===============================================================
# Joint Augmentation for Segmentation (Image + Mask together)
# ===============================================================

class DeepGlobeJointAugmentation:
    """
    Joint augmentation for DeepGlobe that applies the same geometric
    transforms to both the image and the mask.
    """

    def __init__(self, image_size: int = 512):
        self.image_size = image_size
        self.flip = SegmentationRandomFlip(horizontal=True, vertical=True)
        self.rotate = SegmentationRandomRotation(degrees=180, p=0.5)
        self.crop = SegmentationRandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.6, 1.0),
            ratio=(1.0, 1.0),
        )

    def __call__(self, img, mask):
        # Geometric augmentations (applied to both)
        img, mask = self.flip(img, mask)
        img, mask = self.rotate(img, mask)

        # Random crop with 50% probability
        if random.random() > 0.5:
            img, mask = self.crop(img, mask)

        return img, mask


class CoralscapesJointAugmentation:
    """
    Joint augmentation for Coralscapes that applies the same geometric
    transforms to both the image and the mask.
    
    No vertical flip (gravity matters for coral orientation)
    """

    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.flip = SegmentationRandomFlip(horizontal=True, vertical=False)
        self.rotate = SegmentationLimitedRotation(degrees=15, p=0.5)
        self.crop = SegmentationRandomResizedCrop(
            size=image_size,
            scale=(0.7, 1.0),
            ratio=(2.0, 2.0),
        )

    def __call__(self, img, mask):
        # Geometric augmentations (applied to both)
        img, mask = self.flip(img, mask)
        img, mask = self.rotate(img, mask)

        # Random crop with 30% probability
        if random.random() > 0.7:
            img, mask = self.crop(img, mask)

        return img, mask


# =======================
# Factory Functions
# =======================

def get_patchcamelyon_transforms(image_size: int = 224, is_train: bool = True):
    """Get transforms for PatchCamelyon dataset"""
    if is_train:
        return PatchCamelyonAugmentation(image_size=image_size)
    else:
        return T.Compose([
            # T.Resize(image_size),
            # T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


def get_deepglobe_transforms(image_size: int = 512, is_train: bool = True):
    """Get transforms for DeepGlobe dataset (image only, not mask)"""
    if is_train:
        return DeepGlobeAugmentation(image_size=image_size)
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


def get_coralscapes_transforms(image_size: int = 512, is_train: bool = True):
    """Get transforms for Coralscapes dataset (image only, not mask)"""
    if is_train:
        return CoralscapesAugmentation(image_size=image_size)
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


def get_deepglobe_joint_transforms(image_size: int = 512):
    """Get joint transforms for DeepGlobe (for both image and mask)"""
    return DeepGlobeJointAugmentation(image_size=image_size)


def get_coralscapes_joint_transforms(image_size: Tuple[int, int] = (512, 512)):
    """Get joint transforms for Coralscapes (for both image and mask)"""
    return CoralscapesJointAugmentation(image_size=image_size)
