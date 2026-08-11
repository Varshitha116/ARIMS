#!/usr/bin/env python3
"""
Data Preprocessing Module for ARIMS

This module handles data preparation before feeding into the defect detection model:
- Image validation using standard library
- Defect analysis from annotations
- Dataset splitting (train/validation)

All done using only Python standard library - no external dependencies.
"""

import os
import json
import struct
import zlib
import binascii
from pathlib import Path
import shutil


def validate_png_image(filepath):
    """
    Check if a PNG file is valid using only standard library tools.
    Returns dict with validation status and details.
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            if header != b'\x89PNG\r\n\x1a\n':
                return {'valid': False, 'error': 'Invalid PNG signature'}

            # Read IHDR chunk
            f.seek(8)
            chunk_length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)

            if chunk_type == b'IHDR':
                width, height = struct.unpack('>II', f.read(8))
            else:
                # Search for IHDR chunk
                while True:
                    chunk_length = struct.unpack('>I', f.read(4))[0]
                    chunk_type = f.read(4)
                    if chunk_type == b'IHDR':
                        width, height = struct.unpack('>II', f.read(8))
                        break
                    f.read(chunk_length + 4)
                    if chunk_type == b'IEND':
                        return {'valid': False, 'error': 'PNG file incomplete'}

            return {
                'valid': True,
                'width': width,
                'height': height,
                'format': f'PNG_{width}x{height}'
            }

    except Exception as e:
        return {'valid': False, 'error': str(e)}


def extract_defect_stats(annotations_dir):
    """
    Count defects by type across all annotation files.

    Returns: (defect_counts_dict, total_defects_int)
    """
    defect_counts = {}
    total_defects = 0

    for ann_file in Path(annotations_dir).glob('*.json'):
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)

            if 'defects' in data:
                for defect in data['defects']:
                    defect_type = defect.get('type', 'unknown')
                    defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                    total_defects += 1
        except Exception as e:
            print(f"   ⚠️ Could not process {ann_file}: {e}")

    print(f"📊 Defect Statistics:")
    print(f"   Total defects found: {total_defects}")
    for defect_type, count in sorted(defect_counts.items()):
        print(f"   - {defect_type}: {count}")

    return defect_counts, total_defects


def create_train_val_split(images_dir, annotations_dir, output_dir, train_ratio=0.8):
    """
    Split data into training and validation sets.

    Returns: (train_count_int, val_count_int)
    """
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'

    # Create directories
    for d in [train_dir, train_dir/'images', train_dir/'annotations',
              val_dir, val_dir/'images', val_dir/'annotations']:
        d.mkdir(parents=True, exist_ok=True)

    # Get all files
    all_images = sorted(list(Path(images_dir).glob('*.png')))

    # Calculate split
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # Copy training files
    for img in train_images:
        # Copy image
        shutil.copy(img, train_dir / 'images' / img.name)
        # Copy annotation
        ann_src = Path(annotations_dir) / f"{img.stem}.json"
        if ann_src.exists():
            shutil.copy(ann_src, train_dir / 'annotations' / ann_src.name)

    # Copy validation files
    for img in val_images:
        shutil.copy(img, val_dir / 'images' / img.name)
        ann_src = Path(annotations_dir) / f"{img.stem}.json"
        if ann_src.exists():
            shutil.copy(ann_src, val_dir / 'annotations' / ann_src.name)

    print(f"\n✅ Created train/val split:")
    print(f"   Training images: {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")

    return len(train_images), len(val_images)


def run_data_pipeline():
    """
    Run the complete data preprocessing pipeline.
    """
    print("🚀 Starting ARIMS Data Preprocessing Pipeline...\n")

    # Step 1: Validate existing images
    print("📏 Step 1: Validating image files...")
    image_valid = 0
    image_invalid = 0

    for img_path in Path('data/images').glob('*.png'):
        result = validate_png_image(img_path)
        if result['valid']:
            image_valid += 1
            print(f"   ✅ {img_path.name}: Valid PNG")
        else:
            image_invalid += 1
            print(f"   ❌ {img_path.name}: Invalid - {result.get('error', 'Unknown error')}")

    print(f"\n   Total: {image_valid} valid, {image_invalid} invalid")

    # Step 2: Analyze defects
    print("\n📊 Step 2: Analyzing defect distribution...")
    defect_counts, total_defects = extract_defect_stats('data/annotations')

    # Step 3: Split data
    print("\n✂️ Step 3: Creating train/validation split...")
    train_count, val_count = create_train_val_split(
        'data/images',
        'data/annotations',
        'datasets/processed',
        train_ratio=0.8
    )

    # Summary
    print("\n✅ Data Preprocessing Complete!")
    print(f"   Images processed: {image_valid}")
    print(f"   Total defects: {total_defects}")
    print(f"   Train set: {train_count} images")
    print(f"   Validation set: {val_count} images")
    print("\nReady for Milestone 3 (Defect Detection Model)!")

    return {
        'images_processed': image_valid,
        'defects_found': total_defects,
        'train_images': train_count,
        'val_images': val_count
    }


if __name__ == "__main__":
    run_data_pipeline()