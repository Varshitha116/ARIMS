#!/usr/bin/env python3
"""
Agent Orchestrator for ARIMS Multi-Agent System

Central coordinator that manages agent lifecycle, routes messages,
and runs the full detection-to-scheduling pipeline.
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from agents.base_agent import BaseAgent, get_message_bus
from agents.detection_agent import DetectionAgent
from agents.degradation_agent import DegradationAgent
from agents.priority_agent import PriorityAgent
from agents.scheduler_agent import SchedulerAgent
from agents.monitoring_agent import MonitoringAgent


class AgentOrchestrator:
    """
    Central coordinator for the ARIMS multi-agent system.

    Manages:
        1. Agent initialization and lifecycle
        2. Pipeline execution (Detection → Degradation → Priority → Schedule)
        3. System monitoring and health checks
        4. Error recovery and re-planning

    Usage:
        orchestrator = AgentOrchestrator()
        result = orchestrator.run_full_pipeline(image, segment_data)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "yolov8",
        monthly_budget: float = 100000.0,
        daily_capacity: int = 3,
    ):
        self.bus = get_message_bus()

        # Initialize agents
        self.detection_agent = DetectionAgent(
            agent_id="detection_agent_01",
            model_type=model_type,
            model_path=model_path,
        )
        self.degradation_agent = DegradationAgent(
            agent_id="degradation_agent_01"
        )
        self.priority_agent = PriorityAgent(
            agent_id="priority_agent_01"
        )
        self.scheduler_agent = SchedulerAgent(
            agent_id="scheduler_agent_01",
            monthly_budget=monthly_budget,
            daily_capacity=daily_capacity,
        )
        self.monitoring_agent = MonitoringAgent(
            agent_id="monitoring_agent_01"
        )

        # Register all agents with monitoring
        self.monitoring_agent.register_agent(self.detection_agent)
        self.monitoring_agent.register_agent(self.degradation_agent)
        self.monitoring_agent.register_agent(self.priority_agent)
        self.monitoring_agent.register_agent(self.scheduler_agent)
        self.monitoring_agent.register_agent(self.monitoring_agent)

        self._pipeline_runs: List[Dict] = []
        self._created_at = datetime.now().isoformat()

    def run_full_pipeline(
        self,
        image_input: Any,
        segment_data: Optional[Dict] = None,
        road_segments: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete multi-agent pipeline:

        1. Detection Agent: Detect defects in image
        2. Degradation Agent: Predict road segment degradation
        3. Priority Agent: Prioritize repair jobs
        4. Scheduler Agent: Create repair schedule
        5. Monitoring Agent: Track pipeline performance

        Args:
            image_input: Image path, numpy array, or PIL Image
            segment_data: Optional road segment metadata
            road_segments: Optional list of road segment data for degradation

        Returns:
            Complete pipeline result dictionary
        """
        pipeline_start = time.perf_counter()
        pipeline_id = f"pipeline_{int(time.time())}"

        results = {
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "errors": [],
        }

        # ===== Stage 1: Detection =====
        try:
            detection_input = {
                "image": image_input,
                "segment_id": segment_data.get("segment_id", "SEG-001") if segment_data else "SEG-001",
                "location": segment_data.get("location", {}) if segment_data else {},
            }
            detection_report = self.detection_agent.run(detection_input)
            results["stages"]["detection"] = detection_report
            self.monitoring_agent.increment_metric("total_detections")
            self.monitoring_agent.increment_metric(
                "total_defects_found",
                detection_report.get("num_defects", 0)
            )
        except Exception as e:
            results["errors"].append({"stage": "detection", "error": str(e)})
            detection_report = {"num_defects": 0, "severity_score": 0, "defect_summary": {}}
            results["stages"]["detection"] = {"error": str(e)}

        # ===== Stage 2: Degradation Prediction =====
        try:
            if road_segments:
                segments = road_segments
            else:
                # Create default segment from detection results
                severity_score = detection_report.get("severity_score", 0.3)
                defect_count = detection_report.get("num_defects", 0)
                segments = [{
                    "segment_id": segment_data.get("segment_id", "SEG-001") if segment_data else "SEG-001",
                    "current_pci": max(20, 85 - (severity_score * 60) - (defect_count * 3)),
                    "material": segment_data.get("material", "asphalt") if segment_data else "asphalt",
                    "climate": segment_data.get("climate", "temperate") if segment_data else "temperate",
                    "traffic": segment_data.get("traffic", "medium") if segment_data else "medium",
                    "defect_density": defect_count * 2.0,
                }]

            degradation_report = self.degradation_agent.run(segments)
            results["stages"]["degradation"] = degradation_report
        except Exception as e:
            results["errors"].append({"stage": "degradation", "error": str(e)})
            degradation_report = {"assessments": []}
            results["stages"]["degradation"] = {"error": str(e)}

        # ===== Stage 3: Priority Ranking =====
        try:
            # Build job candidates from detection + degradation
            job_candidates = []
            assessments = degradation_report.get("assessments", [])
            det_summary = detection_report.get("defect_summary", {})

            for assessment in assessments:
                job = {
                    "segment_id": assessment.get("segment_id", "SEG-001"),
                    "severity_score": detection_report.get("severity_score", 0.3),
                    "defect_types": list(det_summary.keys()),
                    "traffic": assessment.get("traffic", segment_data.get("traffic", "medium") if segment_data else "medium"),
                    "risk_score": assessment.get("risk_score", 0.3),
                    "estimated_cost": detection_report.get("severity_score", 0.3) * 5000 + 500,
                    "is_highway": segment_data.get("is_highway", False) if segment_data else False,
                }
                job_candidates.append(job)

            if not job_candidates:
                # Create a default job from detection alone
                job_candidates = [{
                    "segment_id": "SEG-001",
                    "severity_score": detection_report.get("severity_score", 0.3),
                    "defect_types": list(det_summary.keys()),
                    "traffic": "medium",
                    "risk_score": 0.3,
                    "estimated_cost": 1500,
                }]

            priority_report = self.priority_agent.run(job_candidates)
            results["stages"]["priority"] = priority_report
        except Exception as e:
            results["errors"].append({"stage": "priority", "error": str(e)})
            priority_report = {"top_5_jobs": []}
            results["stages"]["priority"] = {"error": str(e)}

        # ===== Stage 4: Scheduling =====
        try:
            # Get priority queue
            priority_queue = self.priority_agent.get_priority_queue()
            scheduler_report = self.scheduler_agent.run(priority_queue)
            results["stages"]["scheduling"] = scheduler_report
            self.monitoring_agent.increment_metric("total_jobs_scheduled",
                                                    scheduler_report.get("scheduled_jobs", 0))
        except Exception as e:
            results["errors"].append({"stage": "scheduling", "error": str(e)})
            results["stages"]["scheduling"] = {"error": str(e)}

        # ===== Stage 5: Monitoring =====
        try:
            monitoring_report = self.monitoring_agent.run({})
            results["stages"]["monitoring"] = monitoring_report
        except Exception as e:
            results["errors"].append({"stage": "monitoring", "error": str(e)})

        # ===== Pipeline Summary =====
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        results["total_pipeline_ms"] = round(total_ms, 2)
        results["success"] = len(results["errors"]) == 0

        self._pipeline_runs.append(results)
        return results

    def run_detection_only(self, image_input: Any) -> Dict:
        """Run only the detection stage."""
        return self.detection_agent.run({"image": image_input})

    def run_degradation_only(self, segments: List[Dict]) -> Dict:
        """Run only the degradation prediction stage."""
        return self.degradation_agent.run(segments)

    def run_scheduling_only(self, jobs: List[Dict]) -> Dict:
        """Run priority + scheduling stages."""
        priority_report = self.priority_agent.run(jobs)
        priority_queue = self.priority_agent.get_priority_queue()
        scheduler_report = self.scheduler_agent.run(priority_queue)
        return {
            "priority": priority_report,
            "scheduling": scheduler_report,
        }

    def get_system_status(self) -> Dict:
        """Get full system status."""
        return {
            "orchestrator": {
                "created_at": self._created_at,
                "pipeline_runs": len(self._pipeline_runs),
            },
            "agents": self.monitoring_agent.get_all_agent_statuses(),
            "system_metrics": self.monitoring_agent.get_system_metrics(),
            "alerts": self.monitoring_agent.get_alerts(),
            "message_history": get_message_bus().get_history(limit=20),
        }

    def get_pipeline_history(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline execution history."""
        return self._pipeline_runs[-limit:]

    def get_all_agent_logs(self) -> Dict[str, List[Dict]]:
        """Get logs from all agents."""
        return {
            "detection": self.detection_agent.get_logs(),
            "degradation": self.degradation_agent.get_logs(),
            "priority": self.priority_agent.get_logs(),
            "scheduler": self.scheduler_agent.get_logs(),
            "monitoring": self.monitoring_agent.get_logs(),
        }
