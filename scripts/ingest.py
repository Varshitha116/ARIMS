#!/usr/bin/env python3
"""
Data Ingestion Script for ARIMS Project

Creates sample test data to validate the pipeline infrastructure.
Uses only Python standard library - no external dependencies.
"""

import os
import json
import struct
import zlib
import binascii
from pathlib import Path


def create_png_image(file_path, width=100, height=100):
    """
    Create a simple PNG image using only Python standard library.
    No external image libraries required.
    """
    # PNG file signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk (image header)
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = _create_png_chunk(b'IHDR', ihdr_data)

    # Create raw pixel data (RGB)
    raw_data = b''
    for y in range(height):
        raw_data += b'\x00'  # Filter byte
        for x in range(width):
            # Create a simple pattern
            if (x + y) % 7 == 0:
                raw_data += b'\xff\x32\x32'  # Red defect pattern
            elif y < height // 3:
                raw_data += b'\xc8\xc8\xc8'  # Light gray (top third)
            elif y < 2 * height // 3:
                raw_data += b'\xa0\xa0\xa0'  # Medium gray (middle third)
            else:
                raw_data += b'\x80\x80\x80'  # Dark gray (bottom third)

    # Compress pixel data
    compressed_data = zlib.compress(raw_data)
    idat_chunk = _create_png_chunk(b'IDAT', compressed_data)

    # IEND chunk
    iend_chunk = _create_png_chunk(b'IEND', b'')

    # Write the complete PNG file
    with open(file_path, 'wb') as f:
        f.write(signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)


def _create_png_chunk(chunk_type, data):
    """Create a PNG chunk with CRC checksum"""
    chunk = chunk_type + data
    crc = binascii.crc32(chunk) & 0xffffffff
    return struct.pack('>I', len(data)) + chunk + struct.pack('>I', crc)


def create_directory_structure():
    """Create the required directory structure for the ARIMS data pipeline"""
    print("🔄 Creating directory structure...")

    directories = [
        'datasets/raw',
        'datasets/processed',
        'data/images',
        'data/annotations',
        'data/metadata',
        'models/checkpoints',
        'agents/config'
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_path}")

    print("✅ Directory structure complete\n")


def create_sample_annotation(file_path, num_defects=3):
    """Create a simple annotation file describing road defects"""
    annotations = {
        "filename": Path(file_path).stem + ".png",
        "width": 100,
        "height": 100,
        "defects": [],
        "metadata": {
            "source": "synthetic_test_data",
            "date_created": "2026-08-07",
            "processed": False
        }
    }

    # Create some simple defect annotations
    defect_types = ["crack", "pothole", "surface_damage"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB colors

    for i in range(num_defects):
        defect = {
            "id": i + 1,
            "type": defect_types[i % len(defect_types)],
            "x": (i + 1) * 20,  # X position
            "y": (i + 1) * 15,  # Y position
            "width": 10,
            "height": 10,
            "severity": "medium",
            "color_rgb": colors[i % len(colors)],
            "confidence": 0.85,
            "status": "detected"
        }
        annotations["defects"].append(defect)

    with open(file_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    return file_path


def create_sample_metadata(file_path):
    """Create sample metadata file"""
    metadata = {
        "dataset_source": "synthetic_test_data",
        "total_images": 5,
        "total_defects": 15,
        "defect_types": ["crack", "pothole", "surface_damage"],
        "collection_date": "2026-08-07",
        "location_info": {
            "coordinates": [40.7128, -74.0060],  # NYC coordinates
            "city": "New York",
            "state": "NY",
            "country": "USA"
        },
        "weather_conditions": {
            "temperature": 75,
            "humidity": 65,
            "precipitation": 0,
            "road_surface_temp": 85
        },
        "quality_metrics": {
            "image_resolution": "100x100",
            "defect_detection_rate": 0.85,
            "annotation_completeness": 0.92
        }
    }

    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def run_data_pipeline():
    """Main data pipeline function"""
    print("🚀 Starting ARIMS Data Pipeline Setup\n")

    # Step 1: Create directory structure
    create_directory_structure()

    # Step 2: Create sample data
    print("📝 Creating sample test data (5 images with defects)...")

    for i in range(1, 6):
        image_path = f"data/images/image_{i:03d}.png"
        create_png_image(image_path, width=100, height=100)
        print(f"   Created: {image_path}")

    # Step 3: Create annotations
    print("\n🏷️ Creating defect annotations...")

    for i in range(1, 6):
        annotation_path = f"data/annotations/image_{i:03d}.json"
        defects = 1 + (i % 3)  # Varying number of defects
        create_sample_annotation(annotation_path, num_defects=defects)
        print(f"   Created: {annotation_path} ({defects} defects)")

    # Step 4: Create metadata
    metadata_path = "data/metadata/dataset_metadata.json"
    create_sample_metadata(metadata_path)
    print(f"   Created: {metadata_path}")

    print("\n✅ Sample data creation complete!")


if __name__ == "__main__":
    run_data_pipeline()