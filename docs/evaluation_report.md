# ARIMS Model Evaluation Report

**Generated**: 2026-08-13 15:04:13

---

## Model Comparison

| Model | mAP@0.5 | Precision | Recall | F1 | Latency (ms) |
|---|---|---|---|---|---|
| EfficientDet (Pham 2023) | 0.647 | 0.698 | 0.601 | 0.646 | 68.4 |
| YOLOv5 (Arya et al. 2022) | 0.621 | 0.672 | 0.589 | 0.628 | 45.2 |
| Faster R-CNN (Zhang 2021) | 0.584 | 0.645 | 0.543 | 0.590 | 125.0 |
| SSD MobileNet (RDD2020) | 0.512 | 0.578 | 0.485 | 0.527 | 32.1 |
| **DETR Transformer Detector (Ours)** | **0.006** | **0.011** | **0.443** | **0.021** | **401.8** |
| **YOLOv8 Baseline (Ours)** | **0.000** | **0.000** | **0.000** | **0.000** | **51.8** |


### YOLOv8 Baseline (Ours)

| Metric | Value |
|--------|-------|
| AP@0.5_class_0 | 0.0000 |
| AP@0.5_class_1 | 0.0000 |
| AP@0.5_class_2 | 0.0000 |
| AP@0.5_class_3 | 0.0000 |
| f1 | 0 |
| fps | 19.3000 |
| mAP@0.5 | 0.0000 |
| mAP@0.5:0.95 | 0.0000 |
| mAP@0.75 | 0.0000 |
| mean_ms | 51.7500 |
| precision | 0 |
| recall | 0.0000 |

### DETR Transformer Detector (Ours)

| Metric | Value |
|--------|-------|
| AP@0.5_class_0 | 0.0233 |
| AP@0.5_class_1 | 0.0000 |
| AP@0.5_class_2 | 0.0000 |
| AP@0.5_class_3 | 0.0000 |
| f1 | 0.0213 |
| fps | 2.5000 |
| mAP@0.5 | 0.0058 |
| mAP@0.5:0.95 | 0.0015 |
| mAP@0.75 | 0.0002 |
| mean_ms | 401.8000 |
| precision | 0.0109 |
| recall | 0.4434 |

## Latency Analysis

| Model | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|-------------|----------|----------|
| YOLOv8 Baseline (Ours) | 51.8 | 45.6 | 74.6 | 95.9 |
| DETR Transformer Detector (Ours) | 401.8 | 360.5 | 653.5 | 724.2 |

