#!/usr/bin/env python3
"""Detection API routes for ARIMS."""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import RoadDefectDetector

router = APIRouter()

# Lazy-initialize detector
_detector: Optional[RoadDefectDetector] = None


def get_detector() -> RoadDefectDetector:
    global _detector
    if _detector is None:
        _detector = RoadDefectDetector(model_type="yolov8")
    return _detector


@router.post("/detect")
async def detect_defects(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
):
    """
    Run defect detection on an uploaded road image.

    Returns detected defects with bounding boxes, severity, and repair recommendations.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file temporarily
    suffix = Path(file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        detector = get_detector()
        detector.confidence_threshold = confidence
        result = detector.detect(tmp_path)
        return result.to_dict()
    finally:
        os.unlink(tmp_path)


@router.get("/detections")
async def list_detections(limit: int = Query(20, ge=1, le=100)):
    """List recent detection results (from current session)."""
    # In production, this would query a database
    return {
        "message": "Detection history available via agent orchestrator",
        "hint": "Use POST /api/agents/run-pipeline for full pipeline execution",
    }
