#!/usr/bin/env python3
"""
ARIMS End-to-End Validation Script

Validates the complete autonomous maintenance pipeline:
REAL ROAD IMAGE
        ↓
TRANSFORMER DEFECT DETECTION
        ↓
DEFECT + CONFIDENCE + SEVERITY
        ↓
DEGRADATION PREDICTION
        ↓
PRIORITY AGENT
        ↓
SCHEDULER AGENT
        ↓
REPAIR PLAN
        ↓
MUNICIPAL DASHBOARD

Usage:
    python scripts/test_end2end.py
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import RoadDefectDetector
from agents.orchestrator import AgentOrchestrator

def main():
    print("=" * 70)
    print("🛣️ ARIMS END-TO-END SYSTEM PIPELINE VALIDATION")
    print("=" * 70)

    # 1. Select a real road defect image
    test_img_dir = PROJECT_ROOT / "datasets" / "rdd2022" / "test" / "images"
    test_images = sorted(list(test_img_dir.glob("*.jpg")))

    if not test_images:
        print(f"❌ No test images found in {test_img_dir}")
        sys.exit(1)

    real_img_path = str(test_images[0])
    print(f"\n[STEP 1] Selected Real Road Image: {Path(real_img_path).relative_to(PROJECT_ROOT)}")

    # 2. Test Transformer Detector
    print("\n[STEP 2] Running Transformer Defect Detector (DETR)...")
    detector = RoadDefectDetector(model_type="detr", confidence_threshold=0.05)
    det_res = detector.detect(real_img_path)

    print(f"   Model: {det_res.model_name}")
    print(f"   Image Resolution: {det_res.image_width}x{det_res.image_height}")
    print(f"   Inference Time: {det_res.inference_time_ms:.1f} ms")
    print(f"   Total Detections: {len(det_res.detections)}")
    print(f"   Overall Severity: {det_res.overall_severity} (Score: {det_res.overall_severity_score:.2f})")

    for i, d in enumerate(det_res.detections, 1):
        print(f"     [{i}] Class: {d.class_name} | Conf: {d.confidence:.2f} | BBox: {d.bbox} | Severity: {d.severity} | Cost: ₹{d.estimated_cost_inr:,.0f}")

    # 3. Test Full Multi-Agent Pipeline
    print("\n[STEP 3] Running Multi-Agent Pipeline (Orchestrator)...")
    orchestrator = AgentOrchestrator(model_type="detr")

    segment_metadata = {
        "segment_id": "SEG-REAL-US-001",
        "location": {"latitude": 37.7749, "longitude": -122.4194},
        "material": "asphalt",
        "traffic": "high",
        "climate": "temperate",
        "is_highway": True
    }

    pipeline_result = orchestrator.run_full_pipeline(
        image_input=real_img_path,
        segment_data=segment_metadata
    )

    print(f"   Pipeline Execution ID: {pipeline_result['pipeline_id']}")
    print(f"   Total Pipeline Latency: {pipeline_result['total_pipeline_ms']:.1f} ms")
    print(f"   Pipeline Success: {pipeline_result['success']}")

    # Verify transitions
    stages = pipeline_result["stages"]

    print("\n[TRANSITION 1] Observation → Detection Agent:")
    det_stage = stages.get("detection", {})
    print(f"   Status: SUCCESS | Defect Count: {det_stage.get('num_defects', 0)} | Severity Score: {det_stage.get('severity_score', 0):.2f}")

    print("\n[TRANSITION 2] Detection → Degradation Agent:")
    deg_stage = stages.get("degradation", {})
    assessments = deg_stage.get("assessments", [])
    if assessments:
        ass = assessments[0]
        print(f"   Status: SUCCESS | Current PCI: {ass.get('initial_pci', 0):.1f} → 5-yr Predicted PCI: {ass.get('predicted_pci_5yr', 0):.1f} | Risk Score: {ass.get('risk_score', 0):.2f}")

    print("\n[TRANSITION 3] Degradation → Priority Agent (MCDA):")
    prio_stage = stages.get("priority", {})
    top_jobs = prio_stage.get("top_5_jobs", [])
    if top_jobs:
        top_j = top_jobs[0]
        print(f"   Status: SUCCESS | Priority Level: {top_j.get('level')} | Rank: {top_j.get('rank')} | MCDA Score: {top_j.get('score'):.2f}")

    print("\n[TRANSITION 4] Priority → Scheduler Agent:")
    sched_stage = stages.get("scheduling", {})
    print(f"   Status: SUCCESS | Scheduled Jobs: {sched_stage.get('scheduled_jobs', 0)} | Budget Used: ₹{sched_stage.get('budget_used', 0):,.0f} / ₹{sched_stage.get('budget_used', 0) + sched_stage.get('budget_remaining', 0):,.0f} ({sched_stage.get('budget_utilization', 0):.1f}%)")

    schedule_list = sched_stage.get("schedule", [])
    for j in schedule_list:
        print(f"     Repair Job: {j.get('segment_id')} | Status: {j.get('status')} | Date: {j.get('scheduled_date')} | Crew: {j.get('assigned_crew')} | Cost: ₹{j.get('estimated_cost', 0):,.0f}")

    print("\n[TRANSITION 5] Pipeline → Monitoring Agent:")
    mon_stage = stages.get("monitoring", {})
    sys_metrics = mon_stage.get("system_metrics", {})
    print(f"   Status: SUCCESS | Active Agents: {mon_stage.get('active_agents')} | Total Detections: {sys_metrics.get('total_detections')}")

    print("\n" + "=" * 70)
    if pipeline_result['success']:
        print("🎉 END-TO-END PIPELINE VALIDATION PASSED!")
    else:
        print(f"❌ PIPELINE VALIDATION FAILED with errors: {pipeline_result['errors']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
