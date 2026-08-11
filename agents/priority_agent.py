#!/usr/bin/env python3
"""
Priority Agent for ARIMS Multi-Agent System

Multi-criteria decision analysis (MCDA) agent for prioritizing road repair jobs.
Uses weighted scoring across severity, traffic, degradation rate, cost, and
strategic importance.
"""

import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent


class PriorityAgent(BaseAgent):
    """
    Agent responsible for prioritizing repair jobs using MCDA.

    Criteria weights (configurable):
        - Severity:     0.30  (defect severity score)
        - Safety:       0.25  (public safety impact)
        - Traffic:      0.20  (traffic volume affected)
        - Degradation:  0.15  (degradation acceleration rate)
        - Cost:         0.10  (repair cost efficiency)
    """

    DEFAULT_WEIGHTS = {
        "severity": 0.30,
        "safety": 0.25,
        "traffic": 0.20,
        "degradation": 0.15,
        "cost_efficiency": 0.10,
    }

    TRAFFIC_SCORES = {
        "low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 1.0
    }

    def __init__(
        self,
        agent_id: str = "priority_agent_01",
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "PriorityAgent", config)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._priority_queue: List[Dict] = []

    def perceive(self, input_data: Any) -> Any:
        """
        Collect repair job candidates.

        Args:
            input_data: list of dicts, each with:
                - segment_id: str
                - severity_score: float (0-1)
                - defect_types: list of str
                - traffic: str (low/medium/high/very_high)
                - risk_score: float (0-1) from degradation agent
                - estimated_cost: float (USD)
                - population_density: float (optional, people/km²)
                - is_highway: bool (optional)
        """
        if isinstance(input_data, dict):
            jobs = [input_data]
        elif isinstance(input_data, list):
            jobs = input_data
        else:
            jobs = []

        self._log("Perceive", f"Received {len(jobs)} repair job candidates")
        return jobs

    def analyze(self, observations: Any) -> Any:
        """Score each job across all criteria."""
        jobs = observations
        scored_jobs = []

        for job in jobs:
            scores = {}

            # Severity score (directly from detection)
            scores["severity"] = min(1.0, job.get("severity_score", 0.5))

            # Safety score (based on defect type and location)
            safety_base = scores["severity"]
            if job.get("is_highway", False):
                safety_base *= 1.3
            has_pothole = "D40_Pothole" in job.get("defect_types", [])
            if has_pothole:
                safety_base *= 1.2
            scores["safety"] = min(1.0, safety_base)

            # Traffic impact score
            traffic = job.get("traffic", "medium")
            scores["traffic"] = self.TRAFFIC_SCORES.get(traffic, 0.5)

            # Degradation risk score
            scores["degradation"] = min(1.0, job.get("risk_score", 0.3))

            # Cost efficiency (inverse: lower cost = higher score)
            cost = max(100, job.get("estimated_cost", 1000))
            scores["cost_efficiency"] = min(1.0, 5000 / cost)

            # Weighted composite score
            composite = sum(
                self.weights.get(k, 0) * v for k, v in scores.items()
            )
            composite = min(1.0, composite)

            scored_jobs.append({
                **job,
                "criteria_scores": scores,
                "composite_score": round(composite, 4),
            })

        self._log("Analyze", f"Scored {len(scored_jobs)} jobs")
        return scored_jobs

    def decide(self, analysis: Any) -> Any:
        """Rank jobs by composite score and assign priority levels."""
        scored_jobs = analysis

        # Sort by composite score (highest = most urgent)
        scored_jobs.sort(key=lambda j: j["composite_score"], reverse=True)

        # Assign priority ranks
        for rank, job in enumerate(scored_jobs, 1):
            score = job["composite_score"]
            if score >= 0.75:
                priority_level = "P1_EMERGENCY"
            elif score >= 0.55:
                priority_level = "P2_HIGH"
            elif score >= 0.35:
                priority_level = "P3_MEDIUM"
            else:
                priority_level = "P4_LOW"

            job["priority_rank"] = rank
            job["priority_level"] = priority_level

        self._log("Decide", f"Ranked {len(scored_jobs)} jobs by priority")
        return scored_jobs

    def execute(self, plan: Any) -> Any:
        """Store priority queue and notify downstream agents."""
        self._priority_queue = plan

        # Notify scheduler of priority queue
        self.send_message(
            receiver="scheduler_agent_01",
            msg_type="priority_queue_updated",
            payload={
                "total_jobs": len(plan),
                "p1_count": sum(1 for j in plan if j["priority_level"] == "P1_EMERGENCY"),
                "p2_count": sum(1 for j in plan if j["priority_level"] == "P2_HIGH"),
                "jobs": [
                    {
                        "segment_id": j["segment_id"],
                        "priority_rank": j["priority_rank"],
                        "priority_level": j["priority_level"],
                        "composite_score": j["composite_score"],
                        "estimated_cost": j.get("estimated_cost", 0),
                    }
                    for j in plan
                ]
            },
            priority=1,
        )

        return plan

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate priority ranking report."""
        jobs = results
        return {
            "agent": self.agent_id,
            "total_jobs": len(jobs),
            "priority_distribution": {
                "P1_EMERGENCY": sum(1 for j in jobs if j["priority_level"] == "P1_EMERGENCY"),
                "P2_HIGH": sum(1 for j in jobs if j["priority_level"] == "P2_HIGH"),
                "P3_MEDIUM": sum(1 for j in jobs if j["priority_level"] == "P3_MEDIUM"),
                "P4_LOW": sum(1 for j in jobs if j["priority_level"] == "P4_LOW"),
            },
            "top_5_jobs": [
                {
                    "rank": j["priority_rank"],
                    "segment_id": j["segment_id"],
                    "level": j["priority_level"],
                    "score": j["composite_score"],
                }
                for j in jobs[:5]
            ],
        }

    def get_priority_queue(self) -> List[Dict]:
        """Get current priority queue."""
        return self._priority_queue
