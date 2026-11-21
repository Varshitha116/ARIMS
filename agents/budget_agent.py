#!/usr/bin/env python3
"""
Budget Agent for ARIMS Multi-Agent System

Manages budget allocation across road segments using tiered allocation
strategy. Distributes funds across emergency, planned, and preventive
maintenance tiers using a greedy knapsack approach.
"""

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent


class BudgetAgent(BaseAgent):
    """
    Agent responsible for optimal budget allocation across repair jobs.

    Allocation Tiers:
        - Emergency Reserve:     20% of budget (P1 jobs)
        - Planned Maintenance:   50% of budget (P2/P3 jobs)
        - Preventive Maintenance: 30% of budget (P4 + monitoring)

    Algorithm:
        Greedy knapsack — highest composite_score first within each tier.
    """

    DEFAULT_TIER_SPLIT = {
        "emergency": 0.20,
        "planned": 0.50,
        "preventive": 0.30,
    }

    PRIORITY_TO_TIER = {
        "P1_EMERGENCY": "emergency",
        "P2_HIGH": "planned",
        "P3_MEDIUM": "planned",
        "P4_LOW": "preventive",
    }

    def __init__(
        self,
        agent_id: str = "budget_agent_01",
        total_budget: float = 8_300_000.0,
        tier_split: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "BudgetAgent", config)
        self.total_budget = total_budget
        self.tier_split = tier_split or self.DEFAULT_TIER_SPLIT
        self._allocation: List[Dict] = []
        self._budget_summary: Dict = {}

    # ── Pipeline ──────────────────────────────────────────────

    def perceive(self, input_data: Any) -> Any:
        """
        Receive prioritized job list and optional budget override.

        Args:
            input_data: list of dicts (jobs) OR dict with keys:
                - jobs: list of repair job dicts
                - total_budget: float (optional override)
        """
        if isinstance(input_data, dict):
            jobs = input_data.get("jobs", [])
            budget_override = input_data.get("total_budget")
            if budget_override is not None:
                self.total_budget = float(budget_override)
        elif isinstance(input_data, list):
            jobs = input_data
        else:
            jobs = []

        self._log("Perceive", f"Received {len(jobs)} jobs, budget=INR {self.total_budget:,.0f}")
        return jobs

    def analyze(self, observations: Any) -> Any:
        """Categorize jobs into budget tiers and compute tier budgets."""
        jobs = observations

        # Compute tier budgets
        tier_budgets = {
            tier: round(self.total_budget * fraction, 2)
            for tier, fraction in self.tier_split.items()
        }

        # Sort jobs into tiers
        tiered_jobs: Dict[str, List[Dict]] = {
            "emergency": [],
            "planned": [],
            "preventive": [],
        }

        for job in jobs:
            priority = job.get("priority_level", "P3_MEDIUM")
            tier = self.PRIORITY_TO_TIER.get(priority, "planned")
            tiered_jobs[tier].append(job)

        # Sort each tier by composite_score (highest first)
        for tier in tiered_jobs:
            tiered_jobs[tier].sort(
                key=lambda j: j.get("composite_score", 0), reverse=True
            )

        self._log(
            "Analyze",
            f"Tiers: emergency={len(tiered_jobs['emergency'])}, "
            f"planned={len(tiered_jobs['planned'])}, "
            f"preventive={len(tiered_jobs['preventive'])}"
        )

        return {
            "tier_budgets": tier_budgets,
            "tiered_jobs": tiered_jobs,
        }

    def decide(self, analysis: Any) -> Any:
        """Allocate budget using greedy knapsack within each tier."""
        tier_budgets = analysis["tier_budgets"]
        tiered_jobs = analysis["tiered_jobs"]

        funded: List[Dict] = []
        unfunded: List[Dict] = []
        allocation_by_tier: Dict[str, Dict] = {}

        for tier_name in ["emergency", "planned", "preventive"]:
            budget_remaining = tier_budgets[tier_name]
            tier_funded = []
            tier_unfunded = []

            for job in tiered_jobs[tier_name]:
                cost = job.get("estimated_cost", 0)
                if cost <= budget_remaining:
                    job_alloc = {
                        **job,
                        "funding_status": "FUNDED",
                        "funding_tier": tier_name,
                        "allocated_amount": cost,
                    }
                    tier_funded.append(job_alloc)
                    funded.append(job_alloc)
                    budget_remaining -= cost
                else:
                    job_alloc = {
                        **job,
                        "funding_status": "UNFUNDED",
                        "funding_tier": tier_name,
                        "allocated_amount": 0,
                    }
                    tier_unfunded.append(job_alloc)
                    unfunded.append(job_alloc)

            tier_allocated = tier_budgets[tier_name] - budget_remaining
            allocation_by_tier[tier_name] = {
                "budget": tier_budgets[tier_name],
                "allocated": round(tier_allocated, 2),
                "remaining": round(budget_remaining, 2),
                "funded_jobs": len(tier_funded),
                "unfunded_jobs": len(tier_unfunded),
            }

        total_allocated = sum(t["allocated"] for t in allocation_by_tier.values())
        self._log(
            "Decide",
            f"Funded: {len(funded)}, Unfunded: {len(unfunded)}, "
            f"Allocated: INR {total_allocated:,.0f}"
        )

        return {
            "funded": funded,
            "unfunded": unfunded,
            "allocation_by_tier": allocation_by_tier,
            "total_allocated": round(total_allocated, 2),
            "total_remaining": round(self.total_budget - total_allocated, 2),
        }

    def execute(self, plan: Any) -> Any:
        """Store allocation and notify scheduler."""
        self._allocation = plan["funded"]
        self._budget_summary = {
            "total_budget": self.total_budget,
            "total_allocated": plan["total_allocated"],
            "total_remaining": plan["total_remaining"],
            "allocation_by_tier": plan["allocation_by_tier"],
        }

        # Notify scheduler with funded jobs only
        self.send_message(
            receiver="scheduler_agent_01",
            msg_type="budget_allocation_complete",
            payload={
                "funded_jobs": len(plan["funded"]),
                "unfunded_jobs": len(plan["unfunded"]),
                "total_allocated": plan["total_allocated"],
                "jobs": [
                    {
                        "segment_id": j["segment_id"],
                        "priority_level": j.get("priority_level", ""),
                        "allocated_amount": j["allocated_amount"],
                        "funding_tier": j["funding_tier"],
                    }
                    for j in plan["funded"]
                ],
            },
            priority=1,
        )

        return plan

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate budget allocation report."""
        return {
            "agent": self.agent_id,
            "total_budget": self.total_budget,
            "total_allocated": results["total_allocated"],
            "total_remaining": results["total_remaining"],
            "funded_jobs": len(results["funded"]),
            "unfunded_jobs": len(results["unfunded"]),
            "utilization_pct": round(
                results["total_allocated"] / self.total_budget * 100, 1
            ) if self.total_budget > 0 else 0.0,
            "allocation_by_tier": results["allocation_by_tier"],
        }

    # ── Helpers ───────────────────────────────────────────────

    def get_allocation(self) -> List[Dict]:
        """Get current funded job allocations."""
        return self._allocation

    def get_budget_summary(self) -> Dict:
        """Get budget utilization summary."""
        return self._budget_summary

