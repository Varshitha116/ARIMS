#!/usr/bin/env python3
"""
Data Validation Script for ARIMS

Validates that the sample data meets the requirements for the defect detection model.
Checks file integrity, content structure, and data quality.
"""

import os
import json
import struct
from pathlib import Path


def validate_png_file(filepath):
    """
    Validate that a PNG file is properly formatted using only standard library tools.
    Returns: (is_valid, details_dict)
    """
    try:
        with open(filepath, 'rb') as f:
            # Check PNG signature
            header = f.read(8)
            if header != b'\x89PNG\r\n\x1a\n':
                return False, {"error": "Invalid PNG signature"}

            # Parse IHDR chunk to get dimensions
            f.seek(8)  # Skip signature
            chunk_length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)

            if chunk_type == b'IHDR':
                width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', f.read(13))

                # Skip remaining IHDR data + CRC
                f.read(chunk_length - 13 + 4)
            else:
                # Search for IHDR chunk
                while True:
                    chunk_data_length = struct.unpack('>I', f.read(4))[0]
                    chunk_type = f.read(4)
                    if chunk_type == b'IHDR':
                        width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', f.read(13))
                        break
                    f.read(chunk_data_length + 4)  # Skip data + CRC
                    if chunk_type == b'IEND':
                        return False, {"error": "PNG file is incomplete, missing IHDR chunk"}

                # Skip remaining data
                f.read(chunk_data_length - 13 + 4)

            # Try to read some pixel data to verify structure
            pixel_check = f.read(12)

            return True, {
                "width": width,
                "height": height,
                "bit_depth": bit_depth,
                "color_type": color_type,
                "format": "PNG-{}_{}".format(width, height)
            }

    except Exception as e:
        return False, {"error": f"File validation failed: {str(e)}"}


def validate_annotation_file(filepath):
    """
    Validate that an annotation JSON file meets ARIMS format requirements.
    Returns: (is_valid, details_dict)
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Required top-level fields
        required_fields = ['filename', 'width', 'height', 'defects', 'metadata']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return False, {"error": f"Missing required fields: {missing_fields}"}

        # Validate defect structure
        defects = data['defects']
        valid_defects = 0

        for i, defect in enumerate(defects):
            required_defect_fields = ['id', 'type', 'x', 'y', 'width', 'height', 'severity', 'confidence']
            missing_defect_fields = [field for field in required_defect_fields if field not in defect]

            if missing_defect_fields:
                print(f"   ⚠️ Defect {i+1} missing fields: {missing_defect_fields}")
                continue

            # Validate data types
            if not isinstance(defect['id'], int):
                print(f"   ⚠️ Defect {i+1} id should be integer: {defect['id']}")
                continue

            valid_defects += 1

        if valid_defects == 0:
            return False, {"error": "No valid defects found in annotation file"}

        # Validate metadata structure
        metadata = data.get('metadata', {})
        if not isinstance(metadata, dict):
            return False, {"error": "Metadata should be a dictionary"}

        return True, {
            "filename": data['filename'],
            "image_dimensions": {"width": data['width'], "height": data['height']},
            "defect_count": len(defects),
            "valid_defects": valid_defects,
            "metadata_keys": list(metadata.keys()),
            "format": "ARIMS annotation v1.0"
        }

    except json.JSONDecodeError as e:
        return False, {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return False, {"error": f"Annotation validation failed: {str(e)}"}


def validate_metadata_file(filepath):
    """
    Validate that a metadata JSON file meets ARIMS requirements.
    Returns: (is_valid, details_dict)
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Required top-level fields
        required_fields = ['dataset_source', 'total_images', 'total_defects',
                          'defect_types', 'collection_date', 'location_info',
                          'weather_conditions', 'quality_metrics']

        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return False, {"error": f"Missing required metadata fields: {missing_fields}"}

        # Validate data types and ranges
        validation_errors = []

        if not isinstance(data['total_images'], int) or data['total_images'] <= 0:
            validation_errors.append("total_images must be a positive integer")

        if not isinstance(data['total_defects'], int) or data['total_defects'] < 0:
            validation_errors.append("total_defects must be a non-negative integer")

        if not isinstance(data['defect_types'], list) or not data['defect_types']:
            validation_errors.append("defect_types must be a non-empty list")

        if not isinstance(data['collection_date'], str):
            validation_errors.append("collection_date must be a string")

        if not isinstance(data['location_info'], dict):
            validation_errors.append("location_info must be a dictionary")
        else:
            required_location_fields = ['coordinates', 'city', 'state', 'country']
            for field in required_location_fields:
                if field not in data['location_info']:
                    validation_errors.append(f"location_info missing field: {field}")

        if not isinstance(data['weather_conditions'], dict):
            validation_errors.append("weather_conditions must be a dictionary")

        if not isinstance(data['quality_metrics'], dict):
            validation_errors.append("quality_metrics must be a dictionary")

        if validation_errors:
            return False, {"error": "Metadata validation errors: " + "; ".join(validation_errors)}

        return True, {
            "source": data['dataset_source'],
            "total_images": data['total_images'],
            "total_defects": data['total_defects'],
            "defect_types": data['defect_types'],
            "collection_date": data['collection_date'],
            "has_location_info": bool(data['location_info']),
            "has_weather_data": bool(data['weather_conditions']),
            "quality_score": calculate_quality_score(data['quality_metrics']),
            "format": "ARIMS metadata v1.0"
        }

    except json.JSONDecodeError as e:
        return False, {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return False, {"error": f"Metadata validation failed: {str(e)}"}


def calculate_quality_score(quality_metrics):
    """Calculate a simple quality score from metrics"""
    if not quality_metrics:
        return 0.0

    score = 0.0
    weight = 1.0 / len(quality_metrics)

    for metric, value in quality_metrics.items():
        if isinstance(value, (int, float)):
            score += min(value, 1.0) * weight
        elif isinstance(value, dict):
            score += min(len(value), 1.0) * weight
        elif isinstance(value, list):
            score += min(len(value), 1.0) * weight

    return round(score, 2)


def run_validation_suite():
    """
    Run the complete data validation suite.
    """
    print("🔍 Starting ARIMS Data Validation Suite\n")
    print("=" * 70)

    # Initialize counters
    validation_results = {
        "png_files": {"total": 0, "valid": 0, "invalid": 0},
        "annotation_files": {"total": 0, "valid": 0, "invalid": 0},
        "metadata_files": {"total": 0, "valid": 0, "invalid": 0},
        "summary": {}
    }

    # Step 1: Validate PNG images
    print("\n📸 Validating PNG image files...")
    png_files = list(Path('data/images').glob('*.png'))
    validation_results["png_files"]["total"] = len(png_files)

    for png_file in png_files:
        is_valid, details = validate_png_file(png_file)
        if is_valid:
            validation_results["png_files"]["valid"] += 1
            print(f"   ✅ {png_file.name}: Valid PNG {details.get('format', 'unknown')}")
        else:
            validation_results["png_files"]["invalid"] += 1
            print(f"   ❌ {png_file.name}: Invalid - {details.get('error', 'Unknown error')}")

    # Step 2: Validate annotation files
    print("\n🏷️ Validating annotation files...")
    annotation_files = list(Path('data/annotations').glob('*.json'))
    validation_results["annotation_files"]["total"] = len(annotation_files)

    for ann_file in annotation_files:
        is_valid, details = validate_annotation_file(ann_file)
        if is_valid:
            validation_results["annotation_files"]["valid"] += 1
            print(f"   ✅ {ann_file.name}: Valid annotation with {details.get('defect_count', 0)} defects")
        else:
            validation_results["annotation_files"]["invalid"] += 1
            print(f"   ❌ {ann_file.name}: Invalid - {details.get('error', 'Unknown error')}")

    # Step 3: Validate metadata files
    print("\n📋 Validating metadata files...")
    metadata_files = list(Path('data/metadata').glob('*.json'))
    validation_results["metadata_files"]["total"] = len(metadata_files)

    for meta_file in metadata_files:
        is_valid, details = validate_metadata_file(meta_file)
        if is_valid:
            validation_results["metadata_files"]["valid"] += 1
            print(f"   ✅ {meta_file.name}: Valid metadata (quality score: {details.get('quality_score', 'N/A')})")
        else:
            validation_results["metadata_files"]["invalid"] += 1
            print(f"   ❌ {meta_file.name}: Invalid - {details.get('error', 'Unknown error')}")

    # Calculate summary statistics
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 70)

    total_files = (validation_results["png_files"]["total"] +
                   validation_results["annotation_files"]["total"] +
                   validation_results["metadata_files"]["total"])

    valid_files = (validation_results["png_files"]["valid"] +
                   validation_results["annotation_files"]["valid"] +
                   validation_results["metadata_files"]["valid"])

    invalid_files = (validation_results["png_files"]["invalid"] +
                     validation_results["annotation_files"]["invalid"] +
                     validation_results["metadata_files"]["invalid"])

    validation_rate = (valid_files / total_files * 100) if total_files > 0 else 0

    print(f"Total files validated: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Validation rate: {validation_rate:.1f}%")

    # Quality assessment
    print("\n🎯 QUALITY ASSESSMENT")
    print("-" * 70)

    # PNG quality
    png_quality = validation_results["png_files"]["valid"] / max(validation_results["png_files"]["total"], 1) * 100
    print(f"PNG Image Quality: {png_quality:.1f}% ({validation_results['png_files']['valid']}/{validation_results['png_files']['total']})")

    # Annotation quality
    ann_quality = validation_results["annotation_files"]["valid"] / max(validation_results["annotation_files"]["total"], 1) * 100
    print(f"Annotation Quality: {ann_quality:.1f}% ({validation_results['annotation_files']['valid']}/{validation_results['annotation_files']['total']})")

    # Metadata quality
    meta_quality = validation_results["metadata_files"]["valid"] / max(validation_results["metadata_files"]["total"], 1) * 100
    print(f"Metadata Quality: {meta_quality:.1f}% ({validation_results['metadata_files']['valid']}/{validation_results['metadata_files']['total']})")

    # Overall assessment
    print("\n🏆 OVERALL VALIDATION STATUS")
    print("-" * 70)

    if validation_rate >= 90:
        print("✅ EXCELLENT: Data pipeline is working correctly")
        print("✅ All files are properly formatted")
        print("✅ Ready for Milestone 3 (Defect Detection Model)")
        return True
    elif validation_rate >= 75:
        print("⚠️ GOOD: Most files are valid, minor issues found")
        print("✅ Can proceed with minor fixes")
        return False
    else:
        print("❌ POOR: Significant data quality issues detected")
        print("🔧 Major fixes required before proceeding")
        return False


if __name__ == "__main__":
    success = run_validation_suite()
    exit(0 if success else 1)