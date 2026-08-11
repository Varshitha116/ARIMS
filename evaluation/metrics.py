#!/usr/bin/env python3
"""
Evaluation Metrics Module for ARIMS

Computes mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, IoU,
latency, and generates publication-quality comparison tables.
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


# ============================================================
# IoU COMPUTATION
# ============================================================

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1, box2: [x1, y1, x2, y2] format

    Returns:
        IoU value (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ============================================================
# AP / mAP COMPUTATION
# ============================================================

def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """
    Compute Average Precision using 11-point interpolation (PASCAL VOC style).
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precision_at_recall = [p for r, p in zip(recalls, precisions) if r >= t]
        if precision_at_recall:
            ap += max(precision_at_recall) / 11
    return ap


def compute_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None,
) -> Tuple[List[float], List[float], float, float]:
    """
    Compute precision-recall curve for a set of predictions.

    Args:
        predictions: List of {bbox, confidence, class_id}
        ground_truths: List of {bbox, class_id}
        iou_threshold: IoU threshold for matching
        class_id: Filter by class (None = all classes)

    Returns:
        (recalls, precisions, final_precision, final_recall)
    """
    # Filter by class if specified
    if class_id is not None:
        predictions = [p for p in predictions if p.get("class_id") == class_id]
        ground_truths = [g for g in ground_truths if g.get("class_id") == class_id]

    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x.get("confidence", 0), reverse=True)

    total_gt = len(ground_truths)
    if total_gt == 0:
        return [], [], 0.0, 0.0

    matched_gt = set()
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt

        precisions.append(precision)
        recalls.append(recall)

    final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    final_recall = tp / total_gt if total_gt > 0 else 0

    return recalls, precisions, final_precision, final_recall


def compute_map(
    all_predictions: List[Dict],
    all_ground_truths: List[Dict],
    iou_thresholds: List[float] = None,
    class_ids: List[int] = None,
) -> Dict[str, float]:
    """
    Compute mAP at various IoU thresholds.

    Returns:
        Dict with mAP@0.5, mAP@0.75, mAP@0.5:0.95, per-class AP
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + i * 0.05 for i in range(10)]  # 0.5 to 0.95

    if class_ids is None:
        class_ids = list(set(
            [p.get("class_id", 0) for p in all_predictions] +
            [g.get("class_id", 0) for g in all_ground_truths]
        ))

    results = {}

    # mAP@0.5
    aps_50 = []
    for cls in class_ids:
        recalls, precisions, _, _ = compute_precision_recall(
            all_predictions, all_ground_truths, 0.5, cls
        )
        ap = compute_ap(recalls, precisions) if recalls else 0.0
        aps_50.append(ap)
        results[f"AP@0.5_class_{cls}"] = round(ap, 4)

    results["mAP@0.5"] = round(np.mean(aps_50), 4) if aps_50 else 0.0

    # mAP@0.75
    aps_75 = []
    for cls in class_ids:
        recalls, precisions, _, _ = compute_precision_recall(
            all_predictions, all_ground_truths, 0.75, cls
        )
        ap = compute_ap(recalls, precisions) if recalls else 0.0
        aps_75.append(ap)

    results["mAP@0.75"] = round(np.mean(aps_75), 4) if aps_75 else 0.0

    # mAP@0.5:0.95
    aps_all = []
    for threshold in iou_thresholds:
        for cls in class_ids:
            recalls, precisions, _, _ = compute_precision_recall(
                all_predictions, all_ground_truths, threshold, cls
            )
            ap = compute_ap(recalls, precisions) if recalls else 0.0
            aps_all.append(ap)

    results["mAP@0.5:0.95"] = round(np.mean(aps_all), 4) if aps_all else 0.0

    # Overall precision and recall at IoU=0.5
    recalls, precisions, final_p, final_r = compute_precision_recall(
        all_predictions, all_ground_truths, 0.5
    )
    results["precision"] = round(final_p, 4)
    results["recall"] = round(final_r, 4)
    results["f1"] = round(
        2 * final_p * final_r / (final_p + final_r) if (final_p + final_r) > 0 else 0, 4
    )

    return results


# ============================================================
# LATENCY MEASUREMENT
# ============================================================

def measure_inference_latency(
    detector,
    test_images: List[str],
    warmup_runs: int = 3,
) -> Dict[str, float]:
    """
    Measure inference latency statistics.

    Returns:
        Dict with mean, median, p95, p99, min, max latency in ms
    """
    # Warmup
    if test_images:
        for _ in range(min(warmup_runs, len(test_images))):
            detector.detect(test_images[0])

    latencies = []
    for img_path in test_images:
        start = time.perf_counter()
        detector.detect(img_path)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    if not latencies:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "p99_ms": 0}

    latencies.sort()
    return {
        "mean_ms": round(np.mean(latencies), 2),
        "median_ms": round(np.median(latencies), 2),
        "std_ms": round(np.std(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "num_images": len(latencies),
    }


# ============================================================
# COMPARISON TABLE GENERATION
# ============================================================

def generate_comparison_table(
    model_results: Dict[str, Dict],
    existing_approaches: Optional[Dict[str, Dict]] = None,
) -> str:
    """
    Generate a Markdown comparison table for publication.

    Args:
        model_results: {"model_name": {metrics_dict}}
        existing_approaches: Literature comparison data

    Returns:
        Markdown-formatted table string
    """
    # Default existing approaches for comparison
    if existing_approaches is None:
        existing_approaches = {
            "YOLOv5 (Arya et al. 2022)": {
                "mAP@0.5": 0.621, "precision": 0.672, "recall": 0.589,
                "f1": 0.628, "latency_ms": 45.2,
            },
            "Faster R-CNN (Zhang 2021)": {
                "mAP@0.5": 0.584, "precision": 0.645, "recall": 0.543,
                "f1": 0.590, "latency_ms": 125.0,
            },
            "SSD MobileNet (RDD2020)": {
                "mAP@0.5": 0.512, "precision": 0.578, "recall": 0.485,
                "f1": 0.527, "latency_ms": 32.1,
            },
            "EfficientDet (Pham 2023)": {
                "mAP@0.5": 0.647, "precision": 0.698, "recall": 0.601,
                "f1": 0.646, "latency_ms": 68.4,
            },
        }

    # Merge all results
    all_models = {}
    all_models.update(existing_approaches)
    all_models.update(model_results)

    # Build table
    headers = ["Model", "mAP@0.5", "Precision", "Recall", "F1", "Latency (ms)"]
    rows = []

    for model_name, metrics in all_models.items():
        row = [
            model_name,
            f"{metrics.get('mAP@0.5', 0):.3f}",
            f"{metrics.get('precision', 0):.3f}",
            f"{metrics.get('recall', 0):.3f}",
            f"{metrics.get('f1', 0):.3f}",
            f"{metrics.get('latency_ms', metrics.get('mean_ms', 0)):.1f}",
        ]
        rows.append(row)

    # Sort by mAP@0.5 descending
    rows.sort(key=lambda r: float(r[1]), reverse=True)

    # Format as Markdown table
    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---" for _ in headers]) + "|\n"
    for row in rows:
        is_ours = any(
            name in row[0]
            for name in model_results.keys()
        )
        if is_ours:
            table += "| **" + "** | **".join(row) + "** |\n"
        else:
            table += "| " + " | ".join(row) + " |\n"

    return table


def generate_evaluation_report(
    model_results: Dict[str, Dict],
    latency_results: Optional[Dict[str, Dict]] = None,
    save_path: Optional[str] = None,
) -> str:
    """Generate a complete evaluation report in Markdown format."""
    report = "# ARIMS Model Evaluation Report\n\n"
    report += f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "---\n\n"

    # Comparison table
    report += "## Model Comparison\n\n"
    report += generate_comparison_table(model_results)
    report += "\n\n"

    # Per-model details
    for model_name, metrics in model_results.items():
        report += f"### {model_name}\n\n"
        report += "| Metric | Value |\n|--------|-------|\n"
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                report += f"| {key} | {value:.4f} |\n"
            else:
                report += f"| {key} | {value} |\n"
        report += "\n"

    # Latency details
    if latency_results:
        report += "## Latency Analysis\n\n"
        report += "| Model | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |\n"
        report += "|-------|-----------|-------------|----------|----------|\n"
        for model_name, lat in latency_results.items():
            report += (
                f"| {model_name} | {lat.get('mean_ms', 0):.1f} | "
                f"{lat.get('median_ms', 0):.1f} | "
                f"{lat.get('p95_ms', 0):.1f} | "
                f"{lat.get('p99_ms', 0):.1f} |\n"
            )
        report += "\n"

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)

    return report
