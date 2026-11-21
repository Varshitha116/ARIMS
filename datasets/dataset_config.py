#!/usr/bin/env python3
"""
ARIMS — Multi-Dataset Configuration

Manages dataset configurations for cross-domain road defect detection
across Indian, US, and international road conditions.

Supported Datasets:
    1. RDD2022 India  — Indian roads (potholes, monsoon damage, patch failures)
    2. RDD2022 US     — US road surfaces (4,805 images, 11,014 annotations)
    3. Egypt Roads     — Egyptian road surfaces (3,563 images)

All datasets use the unified RDD2022 class taxonomy:
    D00: Longitudinal Crack
    D10: Transverse Crack
    D20: Alligator Crack
    D40: Pothole
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================
# UNIFIED CLASS TAXONOMY
# ============================================================

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}

# Indian-specific defect subtypes (mapped to RDD2022 classes)
INDIAN_DEFECT_MAPPING = {
    "pothole":             3,   # D40
    "patch_failure":       3,   # D40 (treated as pothole-class)
    "waterlogging_damage": 2,   # D20 (alligator-type cracking from water)
    "alligator_crack":     2,   # D20
    "longitudinal_crack":  0,   # D00
    "transverse_crack":    1,   # D10
    "raveling":            2,   # D20 (surface disintegration → alligator class)
    "edge_break":          0,   # D00
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset split."""
    name: str
    region: str
    description: str
    base_path: str
    image_dir: str = "images"
    annotation_dir: str = "annotations"
    annotation_format: str = "pascal_voc"  # pascal_voc | coco | yolo
    num_classes: int = 4
    class_names: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    image_size: int = 640
    # Dataset statistics (filled after validation)
    num_images: int = 0
    num_annotations: int = 0
    # Indian-specific
    climate_zone: str = "temperate"
    has_monsoon_damage: bool = False
    has_waterlogging: bool = False

    def get_image_path(self) -> Path:
        return Path(self.base_path) / self.image_dir

    def get_annotation_path(self) -> Path:
        return Path(self.base_path) / self.annotation_dir

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "region": self.region,
            "description": self.description,
            "num_images": self.num_images,
            "num_annotations": self.num_annotations,
            "num_classes": self.num_classes,
            "climate_zone": self.climate_zone,
            "annotation_format": self.annotation_format,
        }


# ============================================================
# PRE-CONFIGURED DATASETS
# ============================================================

DATASETS: Dict[str, DatasetConfig] = {
    "rdd2022_india": DatasetConfig(
        name="RDD2022 India",
        region="India",
        description=(
            "Indian road surface images from RDD2022 challenge. "
            "Captures potholes, monsoon-induced cracking, patch failures, "
            "and waterlogging damage common on Indian highways and urban roads."
        ),
        base_path="datasets/rdd2022/india",
        annotation_format="pascal_voc",
        num_images=7706,
        num_annotations=17547,
        climate_zone="tropical",
        has_monsoon_damage=True,
        has_waterlogging=True,
    ),
    "rdd2022_us": DatasetConfig(
        name="RDD2022 US",
        region="United States",
        description=(
            "US road surface images from RDD2022 challenge. "
            "4,805 images with 11,014 annotations covering cracks and potholes."
        ),
        base_path="datasets/rdd2022/us",
        annotation_format="pascal_voc",
        num_images=4805,
        num_annotations=11014,
        climate_zone="temperate",
    ),
    "egypt_roads": DatasetConfig(
        name="Egypt Road Surface",
        region="Egypt",
        description=(
            "Egyptian road surface dataset with 3,563 images. "
            "Arid climate conditions with heat-induced cracking patterns."
        ),
        base_path="datasets/egypt",
        annotation_format="pascal_voc",
        num_images=3563,
        num_annotations=0,  # To be filled after download
        climate_zone="arid",
    ),
}


# ============================================================
# MULTI-DATASET MANAGER
# ============================================================

class DatasetManager:
    """
    Manages multiple road defect datasets for cross-domain evaluation.

    Usage:
        mgr = DatasetManager()
        mgr.add_dataset("rdd2022_india")
        mgr.add_dataset("rdd2022_us")
        stats = mgr.get_combined_stats()
    """

    def __init__(self):
        self._active_datasets: Dict[str, DatasetConfig] = {}

    def add_dataset(self, key: str) -> Optional[DatasetConfig]:
        """Register a dataset for use in training/evaluation."""
        if key in DATASETS:
            self._active_datasets[key] = DATASETS[key]
            return DATASETS[key]
        return None

    def get_dataset(self, key: str) -> Optional[DatasetConfig]:
        """Get a specific dataset config."""
        return self._active_datasets.get(key) or DATASETS.get(key)

    def list_datasets(self) -> List[Dict]:
        """List all available datasets."""
        return [ds.to_dict() for ds in DATASETS.values()]

    def list_active(self) -> List[Dict]:
        """List actively registered datasets."""
        return [ds.to_dict() for ds in self._active_datasets.values()]

    def get_combined_stats(self) -> Dict:
        """Get combined statistics across all active datasets."""
        total_images = sum(ds.num_images for ds in self._active_datasets.values())
        total_annotations = sum(ds.num_annotations for ds in self._active_datasets.values())
        regions = list(set(ds.region for ds in self._active_datasets.values()))

        return {
            "active_datasets": len(self._active_datasets),
            "total_images": total_images,
            "total_annotations": total_annotations,
            "regions": regions,
            "num_classes": 4,
            "class_names": CLASS_NAMES,
            "datasets": self.list_active(),
        }

    def validate_paths(self) -> Dict[str, bool]:
        """Check which dataset paths actually exist on disk."""
        return {
            key: ds.get_image_path().exists()
            for key, ds in self._active_datasets.items()
        }


# ============================================================
# CROSS-DOMAIN EVALUATION CONFIG
# ============================================================

CROSS_DOMAIN_EVAL = {
    "description": (
        "Cross-domain generalization study: Train on source domain, "
        "evaluate on target domain to measure detector robustness "
        "across different road surface conditions and climates."
    ),
    "experiments": [
        {
            "name": "US → India Transfer",
            "train": "rdd2022_us",
            "eval": "rdd2022_india",
            "hypothesis": "US-trained model struggles with monsoon damage patterns",
        },
        {
            "name": "India → US Transfer",
            "train": "rdd2022_india",
            "eval": "rdd2022_us",
            "hypothesis": "India-trained model may over-predict potholes on US roads",
        },
        {
            "name": "Combined → Egypt Transfer",
            "train": ["rdd2022_india", "rdd2022_us"],
            "eval": "egypt_roads",
            "hypothesis": "Combined training improves generalization to unseen arid domain",
        },
    ],
}

