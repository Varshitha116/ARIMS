#!/usr/bin/env python3
"""
RT-DETR Training Script for ARIMS

Trains RT-DETR (Real-Time Detection Transformer) for comparative analysis.

Usage:
    python training/train_rtdetr.py --data datasets/rdd2022/rdd2022.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR on RDD2022")
    parser.add_argument("--data", type=str, default="datasets/rdd2022/rdd2022.yaml")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt",
                        help="RT-DETR variant (rtdetr-l/rtdetr-x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=str, default="models/checkpoints")
    parser.add_argument("--name", type=str, default="rtdetr_rdd2022")
    args = parser.parse_args()

    try:
        from ultralytics import RTDETR
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("🚀 ARIMS RT-DETR Training")
    print("=" * 60)
    print(f"   Dataset: {args.data}")
    print(f"   Model:   {args.model}")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   ImgSz:   {args.imgsz}")
    print(f"   Device:  {args.device}")
    print("=" * 60)

    model = RTDETR(args.model)

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
    )

    print("\n✅ RT-DETR training complete!")
    print(f"   Best model: {args.project}/{args.name}/weights/best.pt")

    print("\n📊 Running validation...")
    val_results = model.val()
    print(f"   mAP@0.5:    {val_results.box.map50:.4f}")
    print(f"   mAP@0.5:95: {val_results.box.map:.4f}")


if __name__ == "__main__":
    main()
