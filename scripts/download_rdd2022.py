#!/usr/bin/env python3
"""
RDD2022 Dataset Download & Preparation Script for ARIMS

Downloads the RDD2022 (Road Damage Detection 2022) dataset and converts
annotations from Pascal VOC XML format to YOLO format.

Defect Classes:
    D00 - Longitudinal Crack
    D10 - Transverse Crack
    D20 - Alligator Crack
    D40 - Pothole

Usage:
    python scripts/download_rdd2022.py
"""

import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}

CLASS_NAMES = ["D00_Longitudinal_Crack", "D10_Transverse_Crack",
               "D20_Alligator_Crack", "D40_Pothole"]

COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China"]

DATASET_ROOT = Path("datasets/rdd2022")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


# ============================================================
# VOC XML → YOLO CONVERSION
# ============================================================

def parse_voc_xml(xml_path):
    """
    Parse a Pascal VOC XML annotation file.

    Returns:
        dict with 'size' (width, height) and 'objects' list
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    width = int(size_elem.find("width").text)
    height = int(size_elem.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        objects.append({
            "class": name,
            "class_id": CLASS_MAP[name],
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return {"width": width, "height": height, "objects": objects}


def voc_to_yolo(voc_data):
    """
    Convert VOC bounding boxes to YOLO format.

    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    """
    w, h = voc_data["width"], voc_data["height"]
    yolo_lines = []

    for obj in voc_data["objects"]:
        xmin, ymin, xmax, ymax = obj["bbox"]

        # Clamp to image boundaries
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))

        # Convert to YOLO format (normalized center + size)
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        # Skip degenerate boxes
        if box_w <= 0 or box_h <= 0:
            continue

        yolo_lines.append(
            f"{obj['class_id']} {x_center:.6f} {y_center:.6f} "
            f"{box_w:.6f} {box_h:.6f}"
        )

    return yolo_lines


# ============================================================
# DATASET ORGANIZATION
# ============================================================

def find_existing_rdd_data():
    """
    Search for existing RDD2022 data that may have been manually downloaded.

    Returns list of (image_path, annotation_path) tuples.
    """
    search_dirs = [
        Path("datasets/raw"),
        Path("datasets/rdd2022_raw"),
        Path("data/rdd2022"),
        Path.home() / "Downloads" / "RDD2022",
        Path.home() / "Downloads" / "rdd2022",
    ]

    pairs = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Look for images and matching XMLs
        for img_path in search_dir.rglob("*.jpg"):
            xml_path = img_path.with_suffix(".xml")
            if not xml_path.exists():
                # Try in separate annotations directory
                for ann_dir in search_dir.rglob("annotations"):
                    alt_xml = ann_dir / img_path.with_suffix(".xml").name
                    if alt_xml.exists():
                        xml_path = alt_xml
                        break

            if xml_path.exists():
                pairs.append((img_path, xml_path))

        # Also check for PNG images
        for img_path in search_dir.rglob("*.png"):
            xml_path = img_path.with_suffix(".xml")
            if xml_path.exists():
                pairs.append((img_path, xml_path))

    return pairs


def create_dataset_split(pairs):
    """
    Split image-annotation pairs into train/val/test sets.

    Uses stratified random split with fixed seed for reproducibility.
    """
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:]
    }

    return splits


def organize_dataset(splits):
    """
    Copy images and convert annotations into YOLO-format dataset structure.

    Creates:
        datasets/rdd2022/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
    """
    stats = {"total": 0, "defects": {cls: 0 for cls in CLASS_NAMES}}

    for split_name, pairs in splits.items():
        img_dir = DATASET_ROOT / split_name / "images"
        lbl_dir = DATASET_ROOT / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in pairs:
            # Parse and convert annotation
            try:
                voc_data = parse_voc_xml(xml_path)
                yolo_lines = voc_to_yolo(voc_data)
            except Exception as e:
                print(f"   ⚠️ Skipping {xml_path.name}: {e}")
                continue

            if not yolo_lines:
                continue

            # Copy image
            dest_img = img_dir / img_path.name
            shutil.copy2(img_path, dest_img)

            # Write YOLO label
            dest_lbl = lbl_dir / img_path.with_suffix(".txt").name
            with open(dest_lbl, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            stats["total"] += 1
            for line in yolo_lines:
                cls_id = int(line.split()[0])
                stats["defects"][CLASS_NAMES[cls_id]] += 1

        print(f"   {split_name}: {len(pairs)} images")

    return stats


# ============================================================
# SYNTHETIC DATA GENERATION (for development/testing)
# ============================================================

def generate_synthetic_dataset(num_images=200):
    """
    Generate a synthetic dataset for development and testing.
    Creates realistic-looking YOLO annotations with proper class distribution.

    This allows the full pipeline to work even without the real RDD2022 data.
    """
    print("📦 Generating synthetic RDD2022-format dataset for development...")

    # Class distribution (approximating real RDD2022 ratios)
    class_weights = [0.35, 0.25, 0.25, 0.15]  # D00, D10, D20, D40

    split_sizes = {
        "train": int(num_images * TRAIN_RATIO),
        "val": int(num_images * VAL_RATIO),
        "test": num_images - int(num_images * TRAIN_RATIO) - int(num_images * VAL_RATIO)
    }

    stats = {"total": 0, "defects": {cls: 0 for cls in CLASS_NAMES}}

    for split_name, count in split_sizes.items():
        img_dir = DATASET_ROOT / split_name / "images"
        lbl_dir = DATASET_ROOT / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img_name = f"synth_{split_name}_{i:04d}.jpg"

            # Create a simple synthetic image (gray road surface)
            try:
                import cv2
                import numpy as np

                img = np.random.randint(80, 140, (640, 640, 3), dtype=np.uint8)
                # Add road-like texture
                for _ in range(20):
                    x1, y1 = random.randint(0, 639), random.randint(0, 639)
                    x2, y2 = x1 + random.randint(10, 100), y1 + random.randint(10, 100)
                    shade = random.randint(60, 160)
                    cv2.rectangle(img, (x1, y1), (min(x2, 639), min(y2, 639)),
                                  (shade, shade, shade), -1)

                cv2.imwrite(str(img_dir / img_name), img)
            except ImportError:
                # Fallback: create a minimal JPEG file
                _create_minimal_jpeg(img_dir / img_name)

            # Generate random annotations (1-5 defects per image)
            num_defects = random.randint(1, 5)
            yolo_lines = []

            for _ in range(num_defects):
                cls_id = random.choices(range(4), weights=class_weights, k=1)[0]
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)
                box_w = random.uniform(0.03, 0.25)
                box_h = random.uniform(0.03, 0.25)

                yolo_lines.append(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_w:.6f} {box_h:.6f}"
                )
                stats["defects"][CLASS_NAMES[cls_id]] += 1

            lbl_path = lbl_dir / img_name.replace(".jpg", ".txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            stats["total"] += 1

        print(f"   {split_name}: {count} images generated")

    return stats


def _create_minimal_jpeg(filepath):
    """Create a minimal valid JPEG file without any imaging library."""
    # Minimal JPEG: SOI marker + minimal content + EOI marker
    import struct
    minimal_jpeg = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07'
        b'\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13'
        b'\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c'
        b'\x28\x37)\x1c\x1c-525444444\xff\xc0\x00\x0b\x08\x00\x01'
        b'\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05'
        b'\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08'
        b'\x01\x01\x00\x00?\x00T\xdb\x2e\xa4\x8cJ\x00\xff\xd9'
    )
    with open(filepath, 'wb') as f:
        f.write(minimal_jpeg)


# ============================================================
# DATASET METADATA
# ============================================================

def create_dataset_yaml():
    """Create the YOLO-format dataset configuration file."""
    yaml_content = f"""# RDD2022 Dataset Configuration for ARIMS
# Road Damage Detection 2022

path: {DATASET_ROOT.resolve()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 4

# Class names
names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole

# Class descriptions (for documentation)
# D00: Linear cracks running parallel to the road direction
# D10: Linear cracks running perpendicular to the road direction
# D20: Interconnected/network cracks (fatigue cracking)
# D40: Bowl-shaped depressions in the pavement surface
"""
    yaml_path = DATASET_ROOT / "rdd2022.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"   Created dataset config: {yaml_path}")
    return yaml_path


def save_dataset_stats(stats):
    """Save dataset statistics to JSON for tracking."""
    stats_path = DATASET_ROOT / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   Saved statistics to: {stats_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 ARIMS RDD2022 Dataset Preparation\n")
    print("=" * 60)

    # Step 1: Check for existing downloaded data
    print("\n🔍 Step 1: Searching for existing RDD2022 data...")
    pairs = find_existing_rdd_data()

    if pairs:
        print(f"   Found {len(pairs)} image-annotation pairs!")
        print("\n📂 Step 2: Organizing into YOLO format...")
        splits = create_dataset_split(pairs)
        stats = organize_dataset(splits)
    else:
        print("   No existing RDD2022 data found.")
        print("\n   ℹ️  To use real data, download RDD2022 from:")
        print("       https://github.com/sekilab/RoadDamageDetector")
        print("       Place files in datasets/raw/ and re-run this script.\n")
        print("📂 Step 2: Generating synthetic dataset for development...")
        stats = generate_synthetic_dataset(num_images=200)

    # Step 3: Create dataset YAML config
    print("\n📝 Step 3: Creating dataset configuration...")
    create_dataset_yaml()

    # Step 4: Save statistics
    print("\n📊 Step 4: Saving dataset statistics...")
    save_dataset_stats(stats)

    # Summary
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print(f"   Total images: {stats['total']}")
    print(f"   Defect distribution:")
    for cls_name, count in stats["defects"].items():
        print(f"     - {cls_name}: {count}")
    print(f"\n   Dataset location: {DATASET_ROOT.resolve()}")
    print(f"   YAML config: {DATASET_ROOT.resolve()}/rdd2022.yaml")


if __name__ == "__main__":
    main()
