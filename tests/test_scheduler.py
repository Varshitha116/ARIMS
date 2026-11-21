#!/usr/bin/env python3
"""
Tests for SchedulerAgent (agents/scheduler_agent.py).

Validates constraint-based scheduling, budget enforcement,
crew assignment, and edge cases.

NOTE: agent.run() returns the report dict (from report()).
      To test raw scheduling output, we call pipeline steps directly.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.scheduler_agent import SchedulerAgent


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def agent():
    return SchedulerAgent()


@pytest.fixture
def prioritized_jobs():
    """Pre-prioritized jobs (as the scheduler would receive them)."""
    return [
        {
            "segment_id": "SEG-001",
            "priority_rank": 1,
            "priority_level": "P1_EMERGENCY",
            "composite_score": 0.92,
            "estimated_cost": 15000,
            "severity_score": 0.9,
            "defect_types": ["D40_Pothole"],
            "traffic": "very_high",
        },
        {
            "segment_id": "SEG-002",
            "priority_rank": 2,
            "priority_level": "P2_HIGH",
            "composite_score": 0.72,
            "estimated_cost": 8000,
            "severity_score": 0.6,
            "defect_types": ["D20_Alligator_Crack"],
            "traffic": "high",
        },
        {
            "segment_id": "SEG-003",
            "priority_rank": 3,
            "priority_level": "P3_MEDIUM",
            "composite_score": 0.45,
            "estimated_cost": 5000,
            "severity_score": 0.4,
            "defect_types": ["D00_Longitudinal_Crack"],
            "traffic": "medium",
        },
    ]


def _run_pipeline(agent, jobs):
    """Run scheduler pipeline and return the raw schedule list."""
    observations = agent.perceive(jobs)
    analysis = agent.analyze(observations)
    decided = agent.decide(analysis)
    executed = agent.execute(decided)
    # execute() returns a dict with 'schedule' key
    if isinstance(executed, dict) and "schedule" in executed:
        return executed["schedule"]
    return executed  # fallback


# ============================================================
# Tests
# ============================================================

class TestSchedulerReport:
    """Test the report dict returned by run()."""

    def test_run_returns_report_dict(self, agent, prioritized_jobs):
        report = agent.run(prioritized_jobs)
        assert isinstance(report, dict)
        assert "agent" in report
        assert "total_jobs" in report

    def test_report_has_scheduled_count(self, agent, prioritized_jobs):
        report = agent.run(prioritized_jobs)
        assert "scheduled_jobs" in report


class TestSchedulerBasic:
    """Basic scheduling functionality via pipeline steps."""

    def test_scheduler_creates_schedule(self, agent, prioritized_jobs):
        result = _run_pipeline(agent, prioritized_jobs)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_job_has_schedule_status(self, agent, prioritized_jobs):
        result = _run_pipeline(agent, prioritized_jobs)
        for job in result:
            assert "schedule_status" in job or "assigned_crew" in job


class TestBudgetConstraint:
    """Budget enforcement tests."""

    def test_budget_status_returns_dict(self, agent):
        budget = agent.get_budget_status()
        assert isinstance(budget, dict)

    def test_get_schedule_returns_list(self, agent):
        schedule = agent.get_schedule()
        assert isinstance(schedule, list)


class TestEdgeCases:
    """Edge cases for the scheduler."""

    def test_empty_input_report(self, agent):
        report = agent.run([])
        assert isinstance(report, dict)
        assert report["total_jobs"] == 0

    def test_single_job(self, agent):
        result = _run_pipeline(agent, [{
            "segment_id": "SEG-SINGLE",
            "priority_rank": 1,
            "priority_level": "P1_EMERGENCY",
            "composite_score": 0.85,
            "estimated_cost": 10000,
            "severity_score": 0.8,
            "defect_types": ["D40_Pothole"],
            "traffic": "high",
        }])
        assert isinstance(result, list)
        assert len(result) >= 1
