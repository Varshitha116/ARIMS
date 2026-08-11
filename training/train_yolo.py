#!/usr/bin/env python3
"""
YOLOv8 Training Script for ARIMS

Trains YOLOv8 on RDD2022 dataset for road defect detection.

Usage:
    python training/train_yolo.py --data datasets/rdd2022/rdd2022.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on RDD2022")
    parser.add_argument("--data", type=str, default="datasets/rdd2022/rdd2022.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Model variant (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, 0 (GPU), mps (Apple Silicon)")
    parser.add_argument("--project", type=str, default="models/checkpoints")
    parser.add_argument("--name", type=str, default="yolov8_rdd2022")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("🚀 ARIMS YOLOv8 Training")
    print("=" * 60)
    print(f"   Dataset: {args.data}")
    print(f"   Model:   {args.model}")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   ImgSz:   {args.imgsz}")
    print(f"   Device:  {args.device}")
    print("=" * 60)

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    print("\n✅ Training complete!")
    print(f"   Best model saved to: {args.project}/{args.name}/weights/best.pt")

    # Validate
    print("\n📊 Running validation...")
    val_results = model.val()
    print(f"   mAP@0.5:    {val_results.box.map50:.4f}")
    print(f"   mAP@0.5:95: {val_results.box.map:.4f}")
    print(f"   Precision:   {val_results.box.mp:.4f}")
    print(f"   Recall:      {val_results.box.mr:.4f}")


if __name__ == "__main__":
    main()
