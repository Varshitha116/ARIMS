# ARIMS Model Evaluation Report

**Generated**: 2026-08-13 14:44:53

---

## Model Comparison

| Model | mAP@0.5 | Precision | Recall | F1 | Latency (ms) |
|---|---|---|---|---|---|
| EfficientDet (Pham 2023) | 0.647 | 0.698 | 0.601 | 0.646 | 68.4 |
| YOLOv5 (Arya et al. 2022) | 0.621 | 0.672 | 0.589 | 0.628 | 45.2 |
| Faster R-CNN (Zhang 2021) | 0.584 | 0.645 | 0.543 | 0.590 | 125.0 |
| SSD MobileNet (RDD2020) | 0.512 | 0.578 | 0.485 | 0.527 | 32.1 |
| **YOLOv8 Baseline (Ours)** | **0.000** | **0.000** | **0.000** | **0.000** | **99.8** |
| **DETR Transformer Detector (Ours)** | **0.000** | **0.000** | **0.000** | **0.000** | **603.2** |


### YOLOv8 Baseline (Ours)

| Metric | Value |
|--------|-------|
| AP@0.5_class_0 | 0.0000 |
| AP@0.5_class_1 | 0.0000 |
| AP@0.5_class_2 | 0.0000 |
| AP@0.5_class_3 | 0.0000 |
| f1 | 0 |
| fps | 10.0000 |
| mAP@0.5 | 0.0000 |
| mAP@0.5:0.95 | 0.0000 |
| mAP@0.75 | 0.0000 |
| mean_ms | 99.7800 |
| precision | 0 |
| recall | 0.0000 |

### DETR Transformer Detector (Ours)

| Metric | Value |
|--------|-------|
| AP@0.5_class_0 | 0.0000 |
| AP@0.5_class_1 | 0.0000 |
| AP@0.5_class_2 | 0.0000 |
| AP@0.5_class_3 | 0.0000 |
| f1 | 0 |
| fps | 1.7000 |
| mAP@0.5 | 0.0000 |
| mAP@0.5:0.95 | 0.0000 |
| mAP@0.75 | 0.0000 |
| mean_ms | 603.1800 |
| precision | 0 |
| recall | 0.0000 |

## Latency Analysis

| Model | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|-------------|----------|----------|
| YOLOv8 Baseline (Ours) | 99.8 | 85.9 | 186.1 | 251.6 |
| DETR Transformer Detector (Ours) | 603.2 | 459.4 | 1255.1 | 1729.7 |

