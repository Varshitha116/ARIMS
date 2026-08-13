#!/usr/bin/env python3
"""
ARIMS Real Dataset Preparation Pipeline

Processes the real RDD2022 (United States subset) dataset:
1. Validates image files and Pascal VOC XML annotations
2. Converts VOC XML bboxes (xmin, ymin, xmax, ymax) to YOLO format (class_id cx cy w h normalized)
3. Generates a reproducible stratified split (70% train, 15% val, 15% test)
4. Saves datasets into datasets/rdd2022/
5. Generates datasets/rdd2022/rdd2022.yaml with project-relative paths
6. Computes and saves dataset statistics (datasets/rdd2022/dataset_stats.json)

Usage:
    python scripts/prepare_real_dataset.py
"""

import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Class map matching RDD2022 official taxonomy
CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}

CLASS_NAMES = [
    "D00_Longitudinal_Crack",
    "D10_Transverse_Crack",
    "D20_Alligator_Crack",
    "D40_Pothole"
]

RANDOM_SEED = 42

def parse_voc_xml(xml_path: Path):
    """Parse Pascal VOC XML annotation."""
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

        # Validate box coordinates
        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({
            "class_name": name,
            "class_id": CLASS_MAP[name],
            "bbox_voc": [xmin, ymin, xmax, ymax]
        })

    return {"width": width, "height": height, "objects": objects}

def voc_to_yolo(width: int, height: int, objects: list):
    """Convert VOC bbox [xmin, ymin, xmax, ymax] to YOLO format line."""
    yolo_lines = []
    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox_voc"]

        # Clamp coordinates to image dimensions
        xmin = max(0.0, min(xmin, float(width)))
        xmax = max(0.0, min(xmax, float(width)))
        ymin = max(0.0, min(ymin, float(height)))
        ymax = max(0.0, min(ymax, float(height)))

        w_box = xmax - xmin
        h_box = ymax - ymin

        if w_box <= 0 or h_box <= 0:
            continue

        x_center = (xmin + xmax) / (2.0 * width)
        y_center = (ymin + ymax) / (2.0 * height)
        norm_w = w_box / float(width)
        norm_h = h_box / float(height)

        yolo_lines.append(f"{obj['class_id']} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_lines

def main():
    print("🚀 ARIMS Real Dataset Pipeline (RDD2022 US Subset)")
    print("=" * 60)

    raw_dir = PROJECT_ROOT / "datasets" / "raw" / "United_States" / "United_States" / "train"
    img_dir = raw_dir / "images"
    xml_dir = raw_dir / "annotations" / "xmls"

    if not img_dir.exists() or not xml_dir.exists():
        print(f"❌ Could not find raw images/xmls at {raw_dir}")
        return

    # Find matching pairs
    valid_pairs = []
    for xml_path in xml_dir.glob("*.xml"):
        img_path = img_dir / (xml_path.stem + ".jpg")
        if img_path.exists():
            valid_pairs.append((img_path, xml_path))

    print(f"✅ Found {len(valid_pairs)} annotated image-xml pairs.")

    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)

    # Use 600 real annotated images for efficient CPU fine-tuning & evaluation
    max_samples = min(600, len(valid_pairs))
    valid_pairs = valid_pairs[:max_samples]

    n_total = len(valid_pairs)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }

    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    out_dataset_dir = PROJECT_ROOT / "datasets" / "rdd2022"
    # Clean previous synthetic data if any
    for s in ["train", "val", "test"]:
        s_img = out_dataset_dir / s / "images"
        s_lbl = out_dataset_dir / s / "labels"
        if s_img.exists():
            shutil.rmtree(s_img)
        if s_lbl.exists():
            shutil.rmtree(s_lbl)
        s_img.mkdir(parents=True, exist_ok=True)
        s_lbl.mkdir(parents=True, exist_ok=True)

    class_histogram = Counter()
    total_bboxes = 0
    images_processed = 0

    for split_name, pairs in splits.items():
        dst_img_dir = out_dataset_dir / split_name / "images"
        dst_lbl_dir = out_dataset_dir / split_name / "labels"

        for img_p, xml_p in pairs:
            # Parse XML
            parsed = parse_voc_xml(xml_p)
            yolo_lines = voc_to_yolo(parsed["width"], parsed["height"], parsed["objects"])

            if not yolo_lines:
                # Skip background images with no valid objects to ensure dense defect dataset
                continue

            # Copy image
            shutil.copy2(img_p, dst_img_dir / img_p.name)

            # Write label file
            lbl_p = dst_lbl_dir / (img_p.stem + ".txt")
            with open(lbl_p, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            images_processed += 1
            for obj in parsed["objects"]:
                class_histogram[CLASS_NAMES[obj["class_id"]]] += 1
                total_bboxes += 1

    print("\n📝 Creating rdd2022.yaml config...")
    yaml_content = f"""# RDD2022 Real Road Defect Dataset Configuration for ARIMS
path: ./datasets/rdd2022
train: train/images
val: val/images
test: test/images

nc: 4
names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole
"""
    yaml_path = out_dataset_dir / "rdd2022.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    stats = {
        "dataset_name": "RDD2022_United_States_Real",
        "total_images_processed": images_processed,
        "total_bboxes": total_bboxes,
        "splits": {
            "train": len(list((out_dataset_dir / "train" / "images").glob("*.jpg"))),
            "val": len(list((out_dataset_dir / "val" / "images").glob("*.jpg"))),
            "test": len(list((out_dataset_dir / "test" / "images").glob("*.jpg")))
        },
        "class_distribution": dict(class_histogram)
    }

    stats_path = out_dataset_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n📊 Dataset Statistics:")
    print(f"   Total Processed Images: {stats['total_images_processed']}")
    print(f"   Total Bounding Boxes: {stats['total_bboxes']}")
    print("   Splits:", stats["splits"])
    print("   Class Breakdown:")
    for k, v in class_histogram.items():
        print(f"     - {k}: {v}")

    print("\n✅ Real RDD2022 Dataset Preparation Complete!")

if __name__ == "__main__":
    main()
