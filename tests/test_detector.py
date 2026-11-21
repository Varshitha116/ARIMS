#!/usr/bin/env python3
"""
Tests for RoadDefectDetector (models/detector.py).

Validates detection output format, severity classification, and
heuristic fallback — WITHOUT requiring model weights or a GPU.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import (
    RoadDefectDetector, Detection, DetectionResult, classify_severity,
    CLASS_NAMES, SEVERITY_THRESHOLDS,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def detector():
    """Create a detector in heuristic/fallback mode (no GPU needed)."""
    return RoadDefectDetector(model_type="yolov8")


@pytest.fixture
def sample_detection_result():
    """A hand-crafted DetectionResult for format testing."""
    return DetectionResult(
        image_path="test_image.jpg",
        image_width=640,
        image_height=640,
        model_name="yolov8_fallback",
        detections=[
            Detection(
                detection_id="det-001",
                class_id=3,
                class_name="D40_Pothole",
                confidence=0.82,
                bbox=[100, 200, 300, 400],
                area_pixels=20000,
                area_ratio=0.05,
                severity="HIGH",
                severity_score=0.75,
                repair_urgency="Immediate patching required",
                estimated_cost_inr=12000,
            ),
            Detection(
                detection_id="det-002",
                class_id=0,
                class_name="D00_Longitudinal_Crack",
                confidence=0.67,
                bbox=[50, 80, 200, 120],
                area_pixels=6000,
                area_ratio=0.015,
                severity="LOW",
                severity_score=0.25,
                repair_urgency="Seal coating within 30 days",
                estimated_cost_inr=3000,
            ),
        ],
        inference_time_ms=25.0,
        total_time_ms=30.0,
    )


# ============================================================
# Tests — DetectionResult format
# ============================================================

class TestDetectionResultFormat:
    """Ensure DetectionResult.to_dict() has the expected schema."""

    def test_to_dict_has_required_keys(self, sample_detection_result):
        d = sample_detection_result.to_dict()
        required = {
            "detections", "num_detections", "image_path",
            "image_size", "model", "overall_severity",
            "overall_severity_score", "timing",
        }
        assert required.issubset(d.keys())

    def test_num_detections_matches_list(self, sample_detection_result):
        d = sample_detection_result.to_dict()
        assert d["num_detections"] == len(d["detections"])
        assert d["num_detections"] == 2

    def test_severity_score_between_0_and_1(self, sample_detection_result):
        d = sample_detection_result.to_dict()
        assert 0.0 <= d["overall_severity_score"] <= 1.0

    def test_timing_fields_present(self, sample_detection_result):
        d = sample_detection_result.to_dict()
        timing = d["timing"]
        assert "inference_ms" in timing
        assert "total_ms" in timing


class TestDetectorInitialisation:
    """Verify detector can be instantiated without crashing."""

    def test_create_detector(self, detector):
        assert detector is not None
        assert detector.confidence_threshold == 0.25

    def test_confidence_threshold_adjustable(self, detector):
        detector.confidence_threshold = 0.5
        assert detector.confidence_threshold == 0.5


# ============================================================
# Tests — classify_severity (module-level function)
# ============================================================

class TestSeverityClassification:
    """Test the classify_severity module-level function."""

    def test_pothole_large_area_is_critical_or_high(self):
        severity, score, urgency, cost = classify_severity(
            class_id=3,       # Pothole
            area_ratio=0.08,  # Large
        )
        assert severity in ("HIGH", "CRITICAL")
        assert score >= 0.5

    def test_small_crack_is_low_or_medium(self):
        severity, score, urgency, cost = classify_severity(
            class_id=0,        # Longitudinal crack
            area_ratio=0.003,  # Small
        )
        assert severity in ("LOW", "MEDIUM")
        assert score < 0.7

    def test_score_always_between_0_and_1(self):
        for class_id in range(4):
            _, score, _, _ = classify_severity(
                class_id=class_id, area_ratio=0.03,
            )
            assert 0.0 <= score <= 1.0

    def test_returns_four_values(self):
        result = classify_severity(class_id=2, area_ratio=0.05)
        assert len(result) == 4

    def test_cost_is_positive(self):
        _, _, _, cost = classify_severity(class_id=3, area_ratio=0.1)
        assert cost > 0
