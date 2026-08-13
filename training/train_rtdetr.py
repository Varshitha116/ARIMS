#!/usr/bin/env python3
"""
DETR Fine-Tuning Script for ARIMS

Fine-tunes facebook/detr-resnet-50 on the RDD2022 dataset using
HuggingFace Transformers. Converts YOLO-format labels to COCO-style
annotations required by DETR.

Usage:
    python training/train_rtdetr.py                         # small test (2 images, 2 steps)
    python training/train_rtdetr.py --epochs 10 --batch 4   # full training

The script:
    1. Reads YOLO-format labels (class_id cx cy w h, normalized)
    2. Converts them to COCO-style (x_min, y_min, w, h in pixels) for DETR
    3. Fine-tunes DETR with the AdamW optimizer
    4. Saves checkpoints only after genuine training steps
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# CONSTANTS derived from the dataset YAML
# ============================================================

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# DATASET: Reads YOLO labels and converts to DETR format
# ============================================================

class RDD2022DetrDataset(torch.utils.data.Dataset):
    """
    Dataset that reads YOLO-format labels and converts them to the
    COCO-style dict format expected by DETR.

    YOLO format per line:  class_id  cx  cy  w  h   (all normalized 0-1)
    DETR expects per image:
        - pixel_values: preprocessed image tensor
        - labels: list of dicts with:
            - class_labels: LongTensor [N]
            - boxes: FloatTensor [N, 4] in COCO format [x_min, y_min, w, h] normalized 0-1
    """

    def __init__(self, images_dir: Path, labels_dir: Path, processor, max_images: int = 0):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.processor = processor

        # Collect image paths that have matching label files
        self.samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.samples.append((img_path, label_path))

        if max_images > 0:
            self.samples = self.samples[:max_images]

        if not self.samples:
            raise FileNotFoundError(
                f"No image+label pairs found in {self.images_dir} / {self.labels_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Parse ALL annotations for this image
        boxes = []
        class_labels = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Validate ranges
                if not (0 <= cls_id < NUM_CLASSES):
                    continue
                if not (0 < w <= 1 and 0 < h <= 1):
                    continue

                # Convert YOLO center format to COCO format [x_min, y_min, w, h]
                # Both remain normalized (0-1) — DETR processor handles the rest
                x_min = cx - w / 2.0
                y_min = cy - h / 2.0

                # Clamp to [0, 1]
                x_min = max(0.0, min(1.0, x_min))
                y_min = max(0.0, min(1.0, y_min))
                w = min(w, 1.0 - x_min)
                h = min(h, 1.0 - y_min)

                # Store as pixel coords for the COCO annotation dict
                boxes.append([
                    x_min * img_w,
                    y_min * img_h,
                    w * img_w,
                    h * img_h,
                ])
                class_labels.append(cls_id)

        # Must have at least one annotation
        if not boxes:
            # Fallback: dummy annotation (background)
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            class_labels = [0]

        # Build COCO-style annotation dict
        annotations = []
        for i, (box, cls) in enumerate(zip(boxes, class_labels)):
            annotations.append({
                "image_id": idx,
                "category_id": cls,
                "bbox": box,  # [x_min, y_min, w, h] in pixels
                "area": box[2] * box[3],
                "iscrowd": 0,
            })

        target = {
            "image_id": idx,
            "annotations": annotations,
        }

        # Process through DETR processor
        encoding = self.processor(
            images=image,
            annotations=[target],
            return_tensors="pt",
        )

        # Remove batch dimension (DataLoader adds it back)
        pixel_values = encoding["pixel_values"].squeeze(0)

        # Extract labels — the processor converts annotations to the format DETR expects
        labels = encoding["labels"][0]

        return pixel_values, labels


def collate_fn(batch):
    """Custom collate that handles variable-length label dicts."""
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ============================================================
# TRAINING LOOP
# ============================================================

def train(args):
    from transformers import DetrForObjectDetection, DetrImageProcessor

    print("🚀 ARIMS DETR Fine-Tuning")
    print("=" * 60)

    # Resolve paths relative to project root
    dataset_root = PROJECT_ROOT / "datasets" / "rdd2022"
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"

    # Verify paths exist
    for p, name in [(train_images, "train/images"), (train_labels, "train/labels"),
                     (val_images, "val/images"), (val_labels, "val/labels")]:
        if not p.exists():
            print(f"❌ Missing: {p}")
            sys.exit(1)
        count = len(list(p.iterdir()))
        print(f"   ✅ {name}: {count} files")

    print(f"   Classes: {NUM_CLASSES} — {list(CLASS_NAMES.values())}")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   LR:      {args.lr}")
    print(f"   Device:  {args.device}")
    print(f"   Max train images: {args.max_train or 'all'}")
    print("=" * 60)

    device = torch.device(args.device)

    # Load pretrained DETR processor and model
    print("\n📥 Loading pretrained DETR model...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,  # We're changing the classification head
    )
    model.to(device)
    model.train()
    print(f"   Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build datasets
    print("\n📦 Building datasets...")
    train_dataset = RDD2022DetrDataset(
        train_images, train_labels, processor, max_images=args.max_train
    )
    val_dataset = RDD2022DetrDataset(
        val_images, val_labels, processor, max_images=args.max_val
    )
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Verify first sample loads correctly
    print("\n🔬 Verifying first sample...")
    try:
        pv, lbl = train_dataset[0]
        print(f"   pixel_values shape: {pv.shape}")
        print(f"   labels keys: {list(lbl.keys())}")
        print(f"   class_labels: {lbl['class_labels']}")
        print(f"   boxes shape: {lbl['boxes'].shape}")
        print(f"   ✅ Sample verification passed")
    except Exception as e:
        print(f"   ❌ Sample verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    print(f"\n🏋️ Starting training ({args.epochs} epochs)...\n")
    best_val_loss = float("inf")
    total_train_steps = 0
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints" / "detr_rdd2022"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            if loss is None:
                print(f"   ⚠️  Epoch {epoch}, batch {batch_idx}: loss is None — skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            total_train_steps += 1

            if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                avg = epoch_loss / epoch_steps if epoch_steps > 0 else 0
                print(f"   Epoch {epoch}/{args.epochs} "
                      f"[{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} avg={avg:.4f}")

        epoch_time = time.perf_counter() - epoch_start
        avg_train_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]
                outputs = model(pixel_values=pixel_values, labels=labels)
                if outputs.loss is not None:
                    val_loss += outputs.loss.item()
                    val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

        print(f"   📊 Epoch {epoch}/{args.epochs}: "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"time={epoch_time:.1f}s")

        # Save best checkpoint
        if avg_val_loss < best_val_loss and total_train_steps > 0:
            best_val_loss = avg_val_loss
            save_path = checkpoint_dir / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"   💾 Saved best model (val_loss={avg_val_loss:.4f}) → {save_path}")

    # Save final model
    if total_train_steps > 0:
        final_path = checkpoint_dir / "final_model"
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        print(f"\n✅ Training complete! {total_train_steps} total steps across {args.epochs} epochs.")
        print(f"   Best model:  {checkpoint_dir / 'best_model'}")
        print(f"   Final model: {final_path}")
    else:
        print("\n❌ No training steps were executed. No checkpoint saved.")
        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DETR on RDD2022")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2 for testing)")
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Max training images (0 = all)")
    parser.add_argument("--max-val", type=int, default=0,
                        help="Max validation images (0 = all)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
