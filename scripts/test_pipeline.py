#!/usr/bin/env python3
"""
ARIMS Data Pipeline Test Script

Comprehensive test suite for the ARIMS data pipeline.
Validates that sample data and infrastructure are correctly set up.

Run with: python3 scripts/test_pipeline.py
"""

import os
import json
import struct
from pathlib import Path
from scripts.ingest import create_png_image, create_sample_annotation


def test_png_creation():
    """Test PNG image creation functionality"""
    print("🔍 Testing PNG image creation...")

    test_path = "data/images/test_image.png"
    create_png_image(test_path, 50, 50)

    # Verify file exists and is readable
    if not os.path.exists(test_path):
        print(f"❌ PNG file not created: {test_path}")
        return False

    # Check if it's a valid PNG (starts with PNG signature)
    with open(test_path, 'rb') as f:
        header = f.read(8)
        if header != b'\x89PNG\r\n\x1a\n':
            print(f"❌ Invalid PNG signature: {test_path}")
            return False

    # Check dimensions (approximate)
    with open(test_path, 'rb') as f:
        f.seek(16)  # Skip signature and IHDR length
        width_bytes = f.read(4)
        height_bytes = f.read(4)

        width = struct.unpack('>I', width_bytes)[0]
        height = struct.unpack('>I', height_bytes)[0]

        if width == 50 and height == 50:
            print(f"✅ PNG validation passed: {test_path} ({width}x{height})")
            return True
        else:
            print(f"❌ Incorrect PNG dimensions: expected 50x50, got {width}x{height}")
            return False


def test_annotation_creation():
    """Test annotation file creation functionality"""
    print("\n🔍 Testing annotation file creation...")

    test_path = "data/annotations/test_annotation.json"

    # Create annotation
    create_sample_annotation(test_path, 2)

    # Verify file exists
    if not os.path.exists(test_path):
        print(f"❌ Annotation file not created: {test_path}")
        return False

    # Try to parse and validate JSON structure
    try:
        with open(test_path, 'r') as f:
            data = json.load(f)

        # Required fields validation
        required_fields = ['filename', 'width', 'height', 'defects']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f"❌ Missing fields in annotation: {missing_fields}")
            return False

        # Check defect count
        defect_count = len(data['defects'])
        if defect_count == 2:
            print(f"✅ Annotation validation passed: {test_path} ({defect_count} defects)")
            return True
        else:
            print(f"❌ Incorrect defect count: expected 2, got {defect_count}")
            return False

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in annotation file: {test_path} - {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading annotation file: {test_path} - {e}")
        return False


def test_directory_structure():
    """Test that all required directories exist"""
    print("\n🔍 Testing directory structure...")

    required_directories = [
        'datasets/raw',
        'datasets/processed',
        'data/images',
        'data/annotations',
        'data/metadata',
        'models/checkpoints',
        'agents/config'
    ]

    all_exist = True
    for directory in required_directories:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Missing directory: {directory}")
            all_exist = False

    return all_exist


def test_sample_data_summary():
    """Create a summary of sample data created"""
    print("\n🔍 Testing sample data summary...")

    # Count files in key directories
    images = list(Path('data/images').glob('*.png'))
    annotations = list(Path('data/annotations').glob('*.json'))
    metadata_files = list(Path('data/metadata').glob('*.json'))

    print(f"📊 Sample Data Summary:")
    print(f"   - Images: {len(images)}")
    print(f"   - Annotations: {len(annotations)}")
    print(f"   - Metadata files: {len(metadata_files)}")

    # Calculate total defects
    total_defects = 0
    for annotation_file in annotations:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            total_defects += len(data.get('defects', []))

    print(f"   - Total defects: {total_defects}")

    # Validate expected counts (based on ingest.py logic)
    expected_images = 5
    expected_annotations = 5
    expected_total_defects = sum(1 + (i % 3) for i in range(1, 6))  # Varying defects per image

    if len(images) == expected_images and len(annotations) == expected_annotations:
        print(f"✅ Sample data counts match expectations")
        return True
    else:
        print(f"❌ Sample data count mismatch: expected {expected_images} images, {expected_annotations} annotations")
        return False


def run_complete_pipeline_test():
    """Run the complete pipeline test suite"""
    print("🚀 Starting ARIMS Data Pipeline Test Suite\n")
    print("=" * 60)

    tests = [
        ("PNG Creation", test_png_creation),
        ("Annotation Creation", test_annotation_creation),
        ("Directory Structure", test_directory_structure),
        ("Sample Data Summary", test_sample_data_summary),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"📊 FINAL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The data pipeline infrastructure is working correctly")
        print("✅ Ready for Milestone 3 (Defect Detection Model)")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please fix the issues above before proceeding")
        return False


if __name__ == "__main__":
    success = run_complete_pipeline_test()
    exit(0 if success else 1)