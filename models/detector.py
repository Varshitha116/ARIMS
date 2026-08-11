#!/usr/bin/env python3
"""
Unified Defect Detection Module for ARIMS

Provides a single interface for running inference with any supported detection
model (YOLOv8, RT-DETR). Handles model loading, preprocessing, inference,
postprocessing, severity classification, and latency measurement.

Usage:
    from models.detector import RoadDefectDetector

    detector = RoadDefectDetector(model_type="yolov8", model_path="yolov8n.pt")
    results = detector.detect("road_image.jpg")
"""

import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


# ============================================================
# DATA CLASSES
# ============================================================

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}

SEVERITY_THRESHOLDS = {
    "area_ratio": {  # defect area / image area
        "LOW": 0.005,
        "MEDIUM": 0.02,
        "HIGH": 0.05,
        "CRITICAL": 0.10,
    },
    "class_severity": {  # inherent severity by defect type
        0: 0.3,   # Longitudinal crack - moderate
        1: 0.3,   # Transverse crack - moderate
        2: 0.6,   # Alligator crack - severe (structural)
        3: 0.8,   # Pothole - very severe (safety hazard)
    }
}


@dataclass
class Detection:
    """Single defect detection result."""
    detection_id: str = ""
    class_id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: List[float] = field(default_factory=list)  # [x1, y1, x2, y2] pixels
    bbox_normalized: List[float] = field(default_factory=list)  # [x1, y1, x2, y2] 0-1
    area_pixels: float = 0.0
    area_ratio: float = 0.0
    severity: str = "LOW"
    severity_score: float = 0.0
    repair_urgency: str = "Monitor"
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
            "bbox_normalized": [round(v, 6) for v in self.bbox_normalized],
            "area_pixels": round(self.area_pixels, 2),
            "area_ratio": round(self.area_ratio, 6),
            "severity": self.severity,
            "severity_score": round(self.severity_score, 4),
            "repair_urgency": self.repair_urgency,
            "estimated_cost_usd": round(self.estimated_cost_usd, 2),
        }


@dataclass
class DetectionResult:
    """Complete detection result for one image."""
    image_path: str = ""
    image_width: int = 0
    image_height: int = 0
    model_name: str = ""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_time_ms: float = 0.0
    defect_summary: Dict[str, int] = field(default_factory=dict)
    overall_severity: str = "LOW"
    overall_severity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_size": [self.image_width, self.image_height],
            "model": self.model_name,
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "timing": {
                "preprocess_ms": round(self.preprocess_time_ms, 2),
                "inference_ms": round(self.inference_time_ms, 2),
                "postprocess_ms": round(self.postprocess_time_ms, 2),
                "total_ms": round(self.total_time_ms, 2),
            },
            "defect_summary": self.defect_summary,
            "overall_severity": self.overall_severity,
            "overall_severity_score": round(self.overall_severity_score, 4),
        }


# ============================================================
# SEVERITY CLASSIFICATION
# ============================================================

def classify_severity(class_id: int, area_ratio: float) -> tuple:
    """
    Classify defect severity based on class type and area ratio.

    Returns: (severity_label, severity_score, urgency, estimated_cost)
    """
    # Combine area-based and class-based severity
    class_weight = SEVERITY_THRESHOLDS["class_severity"].get(class_id, 0.3)
    thresholds = SEVERITY_THRESHOLDS["area_ratio"]

    # Area-based score (0-1)
    if area_ratio >= thresholds["CRITICAL"]:
        area_score = 1.0
    elif area_ratio >= thresholds["HIGH"]:
        area_score = 0.75
    elif area_ratio >= thresholds["MEDIUM"]:
        area_score = 0.5
    elif area_ratio >= thresholds["LOW"]:
        area_score = 0.25
    else:
        area_score = 0.1

    # Combined severity score
    severity_score = 0.4 * area_score + 0.6 * class_weight
    severity_score = min(1.0, severity_score)

    # Classify
    if severity_score >= 0.7:
        severity = "CRITICAL"
        urgency = "Immediate Repair Required"
        cost = 5000 + (area_ratio * 100000)
    elif severity_score >= 0.5:
        severity = "HIGH"
        urgency = "Repair Within 48 Hours"
        cost = 2000 + (area_ratio * 50000)
    elif severity_score >= 0.3:
        severity = "MEDIUM"
        urgency = "Schedule Repair Within 1 Week"
        cost = 500 + (area_ratio * 20000)
    else:
        severity = "LOW"
        urgency = "Monitor Condition"
        cost = 100 + (area_ratio * 5000)

    return severity, severity_score, urgency, cost


# ============================================================
# DETECTOR CLASS
# ============================================================

class RoadDefectDetector:
    """
    Unified road defect detection interface.

    Supports:
        - YOLOv8 (nano, small, medium, large, xlarge)
        - RT-DETR (transformer-based)
        - Fallback mode (OpenCV-based heuristic for demo)

    Args:
        model_type: "yolov8" or "rtdetr"
        model_path: Path to model weights (.pt file)
        confidence_threshold: Minimum detection confidence
        iou_threshold: NMS IoU threshold
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_type: str = "yolov8",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.model_name = ""

        if model_path and HAS_ULTRALYTICS:
            self._load_model(model_path)
        elif HAS_ULTRALYTICS:
            # Auto-discover trained weights
            auto_path = self._find_trained_weights()
            if auto_path:
                self._load_model(auto_path)
            else:
                self.model_name = "fallback_heuristic"
                print("ℹ️  No trained weights found. Using fallback heuristic detector.")
                print("   Train a model with: python training/train_yolo.py")
        else:
            self.model_name = "fallback_heuristic"
            print("⚠️  ultralytics not installed. Using fallback heuristic detector.")

    def _find_trained_weights(self) -> Optional[str]:
        """Auto-discover trained model weights from standard locations."""
        project_root = Path(__file__).resolve().parent.parent
        search_paths = [
            project_root / "models" / "checkpoints" / "yolov8_rdd2022" / "weights" / "best.pt",
            project_root / "runs" / "detect" / "models" / "checkpoints" / "yolov8_rdd2022" / "weights" / "best.pt",
            project_root / "models" / "checkpoints" / "rtdetr_rdd2022" / "weights" / "best.pt",
        ]
        for path in search_paths:
            if path.exists():
                print(f"🔍 Auto-discovered trained weights: {path.name}")
                return str(path)
        return None

    def _load_model(self, model_path: str):
        """Load a YOLO or RT-DETR model."""
        try:
            self.model = YOLO(model_path)
            self.model_name = Path(model_path).stem
            print(f"✅ Loaded model: {self.model_name} ({self.model_type})")
        except Exception as e:
            print(f"⚠️  Failed to load model {model_path}: {e}")
            print("   Falling back to heuristic detector.")
            self.model = None
            self.model_name = "fallback_heuristic"

    def detect(self, image_input, save_annotated: bool = False) -> DetectionResult:
        """
        Run defect detection on an image.

        Args:
            image_input: File path (str/Path), numpy array, or PIL Image
            save_annotated: If True, save annotated image alongside original

        Returns:
            DetectionResult with all detections and metadata
        """
        total_start = time.perf_counter()

        # --- Preprocess ---
        pre_start = time.perf_counter()
        image, image_path = self._load_image(image_input)
        h, w = image.shape[:2]
        preprocess_ms = (time.perf_counter() - pre_start) * 1000

        # --- Inference ---
        inf_start = time.perf_counter()
        if self.model is not None:
            raw_detections = self._run_model_inference(image)
        else:
            raw_detections = self._run_fallback_detection(image)
        inference_ms = (time.perf_counter() - inf_start) * 1000

        # --- Postprocess ---
        post_start = time.perf_counter()
        detections = self._postprocess(raw_detections, w, h)
        postprocess_ms = (time.perf_counter() - post_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # --- Build result ---
        result = DetectionResult(
            image_path=str(image_path) if image_path else "",
            image_width=w,
            image_height=h,
            model_name=self.model_name,
            detections=detections,
            inference_time_ms=inference_ms,
            preprocess_time_ms=preprocess_ms,
            postprocess_time_ms=postprocess_ms,
            total_time_ms=total_ms,
        )

        # Compute summary
        result.defect_summary = self._compute_summary(detections)
        result.overall_severity, result.overall_severity_score = (
            self._compute_overall_severity(detections)
        )

        # Save annotated image if requested
        if save_annotated and image_path:
            self._save_annotated(image, detections, image_path)

        return result

    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Run detection on multiple images."""
        results = []
        for path in image_paths:
            results.append(self.detect(path))
        return results

    # --------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------

    def _load_image(self, image_input):
        """Load image from various input types."""
        image_path = None

        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            # Assume PIL Image
            image = np.array(image_input)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, image_path

    def _run_model_inference(self, image: np.ndarray) -> List[Dict]:
        """Run YOLO/RT-DETR model inference."""
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "confidence": float(boxes.conf[i].cpu()),
                    "class_id": int(boxes.cls[i].cpu()),
                }
                detections.append(det)

        return detections

    def _run_fallback_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Fallback heuristic detection using OpenCV.
        Uses edge detection + contour analysis to find potential defects.
        Not as accurate as YOLO but works without model weights.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Adaptive thresholding for crack detection
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        min_area = (w * h) * 0.001  # Minimum 0.1% of image area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

            # Classify based on shape
            if aspect_ratio > 5:
                class_id = 0  # Longitudinal or transverse crack
                if bw > bh:
                    class_id = 1  # Transverse
            elif area > (w * h) * 0.02:
                class_id = 3  # Pothole (large area)
            else:
                class_id = 2  # Alligator crack (medium, irregular)

            confidence = min(0.85, 0.3 + (area / (w * h)) * 5)

            detections.append({
                "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                "confidence": confidence,
                "class_id": class_id,
            })

        # Limit to top 10 detections by confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[:10]

    def _postprocess(self, raw_detections: List[Dict], img_w: int, img_h: int) -> List[Detection]:
        """Convert raw detections to Detection objects with severity."""
        detections = []
        img_area = img_w * img_h

        for det in raw_detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            confidence = det["confidence"]

            # Calculate area
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]
            area_pixels = box_w * box_h
            area_ratio = area_pixels / img_area if img_area > 0 else 0

            # Normalized bbox
            bbox_norm = [
                bbox[0] / img_w, bbox[1] / img_h,
                bbox[2] / img_w, bbox[3] / img_h,
            ]

            # Severity classification
            severity, severity_score, urgency, cost = classify_severity(
                class_id, area_ratio
            )

            detection = Detection(
                detection_id=str(uuid.uuid4())[:8],
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, f"Unknown_{class_id}"),
                confidence=confidence,
                bbox=bbox,
                bbox_normalized=bbox_norm,
                area_pixels=area_pixels,
                area_ratio=area_ratio,
                severity=severity,
                severity_score=severity_score,
                repair_urgency=urgency,
                estimated_cost_usd=cost,
            )
            detections.append(detection)

        # Sort by severity score (highest first)
        detections.sort(key=lambda d: d.severity_score, reverse=True)
        return detections

    def _compute_summary(self, detections: List[Detection]) -> Dict[str, int]:
        """Count detections per class."""
        summary = {}
        for det in detections:
            summary[det.class_name] = summary.get(det.class_name, 0) + 1
        return summary

    def _compute_overall_severity(self, detections: List[Detection]) -> tuple:
        """Compute overall severity for the image."""
        if not detections:
            return "NONE", 0.0

        max_score = max(d.severity_score for d in detections)
        avg_score = sum(d.severity_score for d in detections) / len(detections)

        # Weighted: 70% worst defect, 30% average
        overall_score = 0.7 * max_score + 0.3 * avg_score

        if overall_score >= 0.7:
            return "CRITICAL", overall_score
        elif overall_score >= 0.5:
            return "HIGH", overall_score
        elif overall_score >= 0.3:
            return "MEDIUM", overall_score
        else:
            return "LOW", overall_score

    def _save_annotated(self, image: np.ndarray, detections: List[Detection], image_path: str):
        """Save annotated image with bounding boxes drawn."""
        annotated = image.copy()

        colors = {
            "CRITICAL": (0, 0, 255),   # Red
            "HIGH": (0, 128, 255),     # Orange
            "MEDIUM": (0, 255, 255),   # Yellow
            "LOW": (0, 255, 0),        # Green
        }

        for det in detections:
            color = colors.get(det.severity, (255, 255, 255))
            x1, y1, x2, y2 = [int(v) for v in det.bbox]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name.split('_')[0]} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated, (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1), color, -1
            )
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        save_path = Path(image_path).parent / f"{Path(image_path).stem}_annotated.jpg"
        cv2.imwrite(str(save_path), annotated)

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection bounding boxes on image and return annotated copy."""
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        colors = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 128, 255),
            "MEDIUM": (0, 255, 255),
            "LOW": (0, 255, 0),
        }

        for det in detections:
            color = colors.get(det.severity, (255, 255, 255))
            x1, y1, x2, y2 = [int(v) for v in det.bbox]

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det.class_name.split('_', 1)[-1]} | {det.severity} | {det.confidence:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated, (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1), color, -1
            )
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA
            )

        return annotated
