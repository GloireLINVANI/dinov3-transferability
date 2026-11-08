"""
DINOv3 Domain Shift Evaluation
==============================
Evaluation of DINOv3 (pretrained on web data) dense features on specialized visual domains
to investigate whether transfer with linear probing is enough.
"""

from .augmentations import (
    PatchCamelyonAugmentation,
    DeepGlobeAugmentation,
    CoralscapesAugmentation,
    get_patchcamelyon_transforms,
    get_deepglobe_transforms,
    get_coralscapes_transforms,
)
from .domain_shift_datasets import (
    download_all_datasets,
    create_dataloaders,
    PatchCamelyonDataset,
    DeepGlobeDataset,
    CoralscapesDataset,
)
from .models import (
    create_model,
    load_dinov2,
    load_dinov3,
    count_parameters,
    DINOClassifier,
    DINOSegmenter,
)
from .training import (
    Trainer,
)
from .visualization import (
    PCAVisualizer,
    compare_models_pca,
    compare_models_cosine_similarity,
    compare_segmentation_maps,
)
