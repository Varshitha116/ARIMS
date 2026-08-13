#!/usr/bin/env python3
"""
Detection Agent for ARIMS Multi-Agent System

Wraps the road defect detection model. Receives images, runs inference,
enriches detections with severity and geolocation, and publishes results
to the message bus for downstream agents.
"""

import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, AgentMessage
from models.detector import RoadDefectDetector, DetectionResult


class DetectionAgent(BaseAgent):
    """
    Agent responsible for defect detection in road images.

    Pipeline:
        perceive: Load and validate image
        analyze:  Run defect detection model
        decide:   Classify severity and urgency per defect
        execute:  Enrich with geolocation (if available)
        report:   Publish results to message bus
    """

    def __init__(
        self,
        agent_id: str = "detection_agent_01",
        model_type: str = "yolov8",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "DetectionAgent", config)
        # For DETR fine-tuned on CPU, set confidence threshold to 0.05
        conf_thresh = 0.05 if model_type.lower() in ["detr", "rtdetr"] and confidence_threshold == 0.25 else confidence_threshold
        self.detector = RoadDefectDetector(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=conf_thresh,
        )
        self._detection_history: List[Dict] = []

    def perceive(self, input_data: Any) -> Any:
        """
        Load image input for detection.

        Args:
            input_data: dict with 'image' (path, array, or PIL), optional 'location'
        """
        if isinstance(input_data, dict):
            image = input_data.get("image")
            location = input_data.get("location", {})
            segment_id = input_data.get("segment_id", "unknown")
        else:
            image = input_data
            location = {}
            segment_id = "unknown"

        self._log("Perceive", f"Image input received for segment {segment_id}")
        return {
            "image": image,
            "location": location,
            "segment_id": segment_id,
        }

    def analyze(self, observations: Any) -> Any:
        """Run defect detection on the image."""
        image = observations["image"]
        result = self.detector.detect(image)
        self._log(
            "Analyze",
            f"Detected {len(result.detections)} defects in {result.inference_time_ms:.1f}ms"
        )
        return {
            "detection_result": result,
            "location": observations["location"],
            "segment_id": observations["segment_id"],
        }

    def decide(self, analysis: Any) -> Any:
        """Classify overall road segment condition based on detections."""
        result: DetectionResult = analysis["detection_result"]

        # Determine action recommendation
        if result.overall_severity == "CRITICAL":
            action = "EMERGENCY_REPAIR"
            priority = 1
        elif result.overall_severity == "HIGH":
            action = "URGENT_REPAIR"
            priority = 2
        elif result.overall_severity == "MEDIUM":
            action = "SCHEDULED_REPAIR"
            priority = 3
        else:
            action = "MONITOR"
            priority = 4

        self._log("Decide", f"Action: {action} (Priority: {priority})")
        return {
            "detection_result": result,
            "action": action,
            "priority": priority,
            "segment_id": analysis["segment_id"],
            "location": analysis["location"],
        }

    def execute(self, plan: Any) -> Any:
        """Enrich detections with location data and store in history."""
        result: DetectionResult = plan["detection_result"]

        enriched = {
            "segment_id": plan["segment_id"],
            "location": plan["location"],
            "action": plan["action"],
            "priority": plan["priority"],
            "detection_result": result.to_dict(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        self._detection_history.append(enriched)

        # Publish to message bus for downstream agents
        self.send_message(
            receiver="",  # broadcast
            msg_type="detection_complete",
            payload=enriched,
            priority=plan["priority"],
        )

        return enriched

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate detection report."""
        det_result = results["detection_result"]
        return {
            "agent": self.agent_id,
            "segment_id": results["segment_id"],
            "action": results["action"],
            "priority": results["priority"],
            "num_defects": det_result["num_detections"],
            "severity": det_result["overall_severity"],
            "severity_score": det_result["overall_severity_score"],
            "defect_summary": det_result["defect_summary"],
            "timing": det_result["timing"],
        }

    def get_detection_history(self, limit: int = 20) -> List[Dict]:
        """Return recent detection history."""
        return self._detection_history[-limit:]
