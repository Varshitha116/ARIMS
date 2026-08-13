#!/usr/bin/env python3
"""
ARIMS Evaluation & Benchmark Script

Evaluates trained detection models (YOLOv8 baseline & DETR Transformer Detector)
on the real RDD2022 test set.

Calculates:
- Precision, Recall, F1
- mAP@0.5, mAP@0.75, mAP@0.5:0.95
- Inference latency statistics (Mean, Median, P95, P99)
- Outputs genuine evaluation report to docs/evaluation_report.md

Usage:
    python evaluation/benchmark.py
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import RoadDefectDetector
from evaluation.metrics import compute_map, measure_inference_latency, generate_evaluation_report

def load_ground_truths(test_dir: Path):
    """Load ground truth labels from YOLO label txt files."""
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    all_gt = []

    for lbl_file in lbl_dir.glob("*.txt"):
        img_file = img_dir / (lbl_file.stem + ".jpg")
        if not img_file.exists():
            continue

        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:]]

                # Convert normalized [cx, cy, w, h] to normalized [x1, y1, x2, y2]
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

                all_gt.append({
                    "image_id": lbl_file.stem,
                    "class_id": cls_id,
                    "bbox": [x1, y1, x2, y2]
                })

    return all_gt

def run_benchmark():
    print("🔬 ARIMS Detection Model Benchmark")
    print("=" * 60)

    test_dir = PROJECT_ROOT / "datasets" / "rdd2022" / "test"
    test_images = sorted(list((test_dir / "images").glob("*.jpg")))

    if not test_images:
        print(f"❌ No test images found in {test_dir / 'images'}")
        return

    print(f"✅ Found {len(test_images)} real test images in {test_dir.relative_to(PROJECT_ROOT)}")

    # Load ground truths
    ground_truths = load_ground_truths(test_dir)
    print(f"✅ Loaded {len(ground_truths)} ground truth defect annotations.")

    model_results = {}
    latency_results = {}

    # 1. Benchmark YOLOv8 Baseline
    print("\n------------------------------------------------------------")
    print("1. Benchmarking YOLOv8 Baseline Detector...")
    try:
        yolo_detector = RoadDefectDetector(model_type="yolov8", confidence_threshold=0.25)
        print(f"   Loaded model: {yolo_detector.model_name}")
        yolo_preds = []

        start_t = time.perf_counter()
        for img_p in test_images:
            res = yolo_detector.detect(str(img_p))
            # Ignore fallback heuristic predictions during neural network evaluation
            if "fallback" in res.model_name:
                continue

            for d in res.detections:
                w_img, h_img = res.image_width, res.image_height
                norm_box = [
                    d.bbox[0] / float(w_img),
                    d.bbox[1] / float(h_img),
                    d.bbox[2] / float(w_img),
                    d.bbox[3] / float(h_img),
                ]
                yolo_preds.append({
                    "image_id": img_p.stem,
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "bbox": norm_box
                })
        yolo_time = (time.perf_counter() - start_t) * 1000

        yolo_metrics = compute_map(yolo_preds, ground_truths)
        yolo_lat = measure_inference_latency(yolo_detector, [str(p) for p in test_images[:50]])
        yolo_metrics["mean_ms"] = yolo_lat["mean_ms"]
        yolo_metrics["fps"] = round(1000.0 / yolo_lat["mean_ms"], 1) if yolo_lat["mean_ms"] > 0 else 0

        model_results["YOLOv8 Baseline (Ours)"] = yolo_metrics
        latency_results["YOLOv8 Baseline (Ours)"] = yolo_lat

        print(f"   yolov8 mAP@0.5: {yolo_metrics.get('mAP@0.5', 0):.4f} | Precision: {yolo_metrics.get('precision', 0):.4f} | Recall: {yolo_metrics.get('recall', 0):.4f}")
        print(f"   yolov8 Latency: {yolo_lat['mean_ms']} ms ({yolo_metrics['fps']} FPS)")

    except Exception as e:
        print(f"⚠️ YOLOv8 benchmark error: {e}")

    # 2. Benchmark DETR Transformer Detector
    print("\n------------------------------------------------------------")
    print("2. Benchmarking DETR Transformer Detector...")
    try:
        # DETR confidence threshold set to 0.05 to capture queries fine-tuned on CPU
        detr_detector = RoadDefectDetector(model_type="detr", confidence_threshold=0.05)
        print(f"   Loaded model: {detr_detector.model_name}")
        detr_preds = []

        start_t = time.perf_counter()
        for img_p in test_images:
            res = detr_detector.detect(str(img_p))
            if "fallback" in res.model_name:
                continue

            for d in res.detections:
                w_img, h_img = res.image_width, res.image_height
                norm_box = [
                    d.bbox[0] / float(w_img),
                    d.bbox[1] / float(h_img),
                    d.bbox[2] / float(w_img),
                    d.bbox[3] / float(h_img),
                ]
                detr_preds.append({
                    "image_id": img_p.stem,
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "bbox": norm_box
                })
        detr_time = (time.perf_counter() - start_t) * 1000

        detr_metrics = compute_map(detr_preds, ground_truths)
        detr_lat = measure_inference_latency(detr_detector, [str(p) for p in test_images[:50]])
        detr_metrics["mean_ms"] = detr_lat["mean_ms"]
        detr_metrics["fps"] = round(1000.0 / detr_lat["mean_ms"], 1) if detr_lat["mean_ms"] > 0 else 0

        model_results["DETR Transformer Detector (Ours)"] = detr_metrics
        latency_results["DETR Transformer Detector (Ours)"] = detr_lat

        print(f"   DETR mAP@0.5: {detr_metrics.get('mAP@0.5', 0):.4f} | Precision: {detr_metrics.get('precision', 0):.4f} | Recall: {detr_metrics.get('recall', 0):.4f}")
        print(f"   DETR Latency: {detr_lat['mean_ms']} ms ({detr_metrics['fps']} FPS)")

    except Exception as e:
        print(f"⚠️ DETR benchmark error: {e}")

    # Generate genuine report file
    report_path = PROJECT_ROOT / "docs" / "evaluation_report.md"
    generate_evaluation_report(model_results, latency_results, save_path=str(report_path))
    print(f"\n✅ Genuine Evaluation Report saved to: {report_path.relative_to(PROJECT_ROOT)}")

    # Save benchmark JSON results
    out_json = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
    bench_data = {
        "dataset": "RDD2022_United_States_Real",
        "num_test_images": len(test_images),
        "num_ground_truth_defects": len(ground_truths),
        "model_results": model_results,
        "latency_results": latency_results
    }
    with open(out_json, "w") as f:
        json.dump(bench_data, f, indent=2)
    print(f"✅ Genuine Benchmark Data saved to: {out_json.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    run_benchmark()
