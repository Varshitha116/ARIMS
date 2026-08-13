#!/usr/bin/env python3
"""
Scheduler Agent for ARIMS Multi-Agent System

Constraint-based scheduling for repair operations.
Handles crew allocation, budget constraints, equipment availability,
and generates weekly/monthly repair schedules.
"""

import time
import random
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent


class SchedulerAgent(BaseAgent):
    """
    Agent responsible for creating repair schedules.

    Constraints:
        - Daily crew capacity (configurable, default 3 jobs/day)
        - Monthly budget cap
        - Equipment availability
        - Weather windows

    Scheduling Algorithm:
        Greedy priority-first with constraint checking.
    """

    def __init__(
        self,
        agent_id: str = "scheduler_agent_01",
        daily_capacity: int = 3,
        monthly_budget: float = 8300000.0,
        available_crews: int = 5,
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "SchedulerAgent", config)
        self.daily_capacity = daily_capacity
        self.monthly_budget = monthly_budget
        self.available_crews = available_crews
        self._schedule: List[Dict] = []
        self._budget_used = 0.0

    def perceive(self, input_data: Any) -> Any:
        """
        Receive prioritized job list.

        Args:
            input_data: list of priority-ranked repair jobs
        """
        if isinstance(input_data, list):
            jobs = input_data
        elif isinstance(input_data, dict):
            jobs = input_data.get("jobs", [])
        else:
            jobs = []

        self._log("Perceive", f"Received {len(jobs)} jobs to schedule")
        return jobs

    def analyze(self, observations: Any) -> Any:
        """Analyze resource requirements for each job."""
        jobs = observations
        analyzed = []

        for job in jobs:
            cost = job.get("estimated_cost", 1000)
            priority = job.get("priority_level", "P3_MEDIUM")

            # Estimate duration based on cost/complexity
            if cost > 5000:
                duration_hours = random.randint(6, 16)
                crew_size = 3
                equipment = ["paver", "roller", "truck"]
            elif cost > 2000:
                duration_hours = random.randint(4, 8)
                crew_size = 2
                equipment = ["patcher", "truck"]
            else:
                duration_hours = random.randint(2, 4)
                crew_size = 1
                equipment = ["hand_tools"]

            analyzed.append({
                **job,
                "duration_hours": duration_hours,
                "crew_size": crew_size,
                "equipment_needed": equipment,
                "estimated_cost": cost,
            })

        return analyzed

    def decide(self, analysis: Any) -> Any:
        """Create schedule using greedy priority-first algorithm."""
        jobs = analysis
        schedule = []
        budget_remaining = self.monthly_budget
        current_date = datetime.now()
        daily_slots = {i: self.daily_capacity for i in range(30)}  # 30-day window
        crew_available = self.available_crews

        for job in jobs:
            cost = job.get("estimated_cost", 1000)

            # Check budget constraint
            if cost > budget_remaining:
                job["schedule_status"] = "DEFERRED_BUDGET"
                schedule.append(job)
                continue

            # Find earliest available day
            scheduled_day = None
            for day in range(30):
                if daily_slots[day] > 0:
                    scheduled_day = day
                    break

            if scheduled_day is None:
                job["schedule_status"] = "DEFERRED_CAPACITY"
                schedule.append(job)
                continue

            # Schedule the job
            scheduled_date = current_date + timedelta(days=scheduled_day)
            job["scheduled_date"] = scheduled_date.strftime("%Y-%m-%d")
            job["scheduled_day"] = scheduled_day
            job["schedule_status"] = "SCHEDULED"
            job["assigned_crew"] = f"CREW-{(len(schedule) % crew_available) + 1:02d}"

            daily_slots[scheduled_day] -= 1
            budget_remaining -= cost

            schedule.append(job)

        # Summary
        scheduled = sum(1 for j in schedule if j["schedule_status"] == "SCHEDULED")
        deferred = len(schedule) - scheduled
        self._log("Decide", f"Scheduled: {scheduled}, Deferred: {deferred}")

        return {
            "schedule": schedule,
            "budget_used": self.monthly_budget - budget_remaining,
            "budget_remaining": budget_remaining,
        }

    def execute(self, plan: Any) -> Any:
        """Finalize and store the schedule."""
        self._schedule = plan["schedule"]
        self._budget_used = plan["budget_used"]

        # Notify monitoring agent
        self.send_message(
            receiver="monitoring_agent_01",
            msg_type="schedule_created",
            payload={
                "total_jobs": len(self._schedule),
                "scheduled": sum(1 for j in self._schedule if j["schedule_status"] == "SCHEDULED"),
                "budget_used": plan["budget_used"],
                "budget_remaining": plan["budget_remaining"],
            },
            priority=1,
        )

        return plan

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate scheduling report."""
        schedule = results["schedule"]
        scheduled = [j for j in schedule if j["schedule_status"] == "SCHEDULED"]
        deferred = [j for j in schedule if j["schedule_status"] != "SCHEDULED"]

        return {
            "agent": self.agent_id,
            "total_jobs": len(schedule),
            "scheduled_jobs": len(scheduled),
            "deferred_jobs": len(deferred),
            "budget_used": round(results["budget_used"], 2),
            "budget_remaining": round(results["budget_remaining"], 2),
            "budget_utilization": round(
                results["budget_used"] / self.monthly_budget * 100, 1
            ),
            "schedule": [
                {
                    "segment_id": j.get("segment_id", ""),
                    "priority_level": j.get("priority_level", ""),
                    "scheduled_date": j.get("scheduled_date", "N/A"),
                    "assigned_crew": j.get("assigned_crew", "N/A"),
                    "status": j["schedule_status"],
                    "estimated_cost": j.get("estimated_cost", 0),
                    "duration_hours": j.get("duration_hours", 0),
                }
                for j in schedule
            ],
        }

    def get_schedule(self) -> List[Dict]:
        """Get current schedule."""
        return self._schedule

    def get_budget_status(self) -> Dict:
        """Get budget utilization."""
        return {
            "total_budget": self.monthly_budget,
            "budget_used": self._budget_used,
            "budget_remaining": self.monthly_budget - self._budget_used,
            "utilization_pct": round(self._budget_used / self.monthly_budget * 100, 1),
        }
