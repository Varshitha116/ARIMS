#!/usr/bin/env python3
"""
Tests for PriorityAgent (agents/priority_agent.py).

Validates MCDA scoring, priority-level assignment, heapq extraction,
and edge cases like empty input.

NOTE: agent.run() returns the report dict (from report()).
      To test raw scoring, we call perceive/analyze/decide/execute directly.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.priority_agent import PriorityAgent


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def agent():
    return PriorityAgent()


@pytest.fixture
def sample_jobs():
    """Three repair jobs with distinct severity levels."""
    return [
        {
            "segment_id": "SEG-HIGH",
            "severity_score": 0.9,
            "defect_types": ["D40_Pothole"],
            "traffic": "very_high",
            "risk_score": 0.8,
            "estimated_cost": 15000,
            "is_highway": True,
        },
        {
            "segment_id": "SEG-MED",
            "severity_score": 0.5,
            "defect_types": ["D00_Longitudinal_Crack"],
            "traffic": "medium",
            "risk_score": 0.4,
            "estimated_cost": 5000,
            "is_highway": False,
        },
        {
            "segment_id": "SEG-LOW",
            "severity_score": 0.15,
            "defect_types": ["D10_Transverse_Crack"],
            "traffic": "low",
            "risk_score": 0.1,
            "estimated_cost": 20000,
            "is_highway": False,
        },
    ]


def _run_pipeline(agent, jobs):
    """Run the agent pipeline steps and return the raw scored+ranked list."""
    observations = agent.perceive(jobs)
    analysis = agent.analyze(observations)
    decided = agent.decide(analysis)
    executed = agent.execute(decided)
    return executed  # This is the raw list of scored/ranked jobs


# ============================================================
# Tests — MCDA Scoring
# ============================================================

class TestMCDAScoring:
    """Validate multi-criteria scoring correctness."""

    def test_higher_severity_gets_higher_score(self, agent, sample_jobs):
        result = _run_pipeline(agent, sample_jobs)
        scores = {j["segment_id"]: j["composite_score"] for j in result}
        assert scores["SEG-HIGH"] > scores["SEG-MED"] > scores["SEG-LOW"]

    def test_composite_score_between_0_and_1(self, agent, sample_jobs):
        result = _run_pipeline(agent, sample_jobs)
        for job in result:
            assert 0.0 <= job["composite_score"] <= 1.0

    def test_weights_sum_to_1(self, agent):
        total = sum(agent.weights.values())
        assert abs(total - 1.0) < 1e-9


class TestPriorityLevels:
    """Validate priority level assignment."""

    def test_high_score_gets_p1_or_p2(self, agent, sample_jobs):
        result = _run_pipeline(agent, sample_jobs)
        high_job = next(j for j in result if j["segment_id"] == "SEG-HIGH")
        assert high_job["priority_level"] in ("P1_EMERGENCY", "P2_HIGH")

    def test_low_score_gets_p3_or_p4(self, agent, sample_jobs):
        result = _run_pipeline(agent, sample_jobs)
        low_job = next(j for j in result if j["segment_id"] == "SEG-LOW")
        assert low_job["priority_level"] in ("P3_MEDIUM", "P4_LOW")

    def test_rank_assignment_is_sequential(self, agent, sample_jobs):
        result = _run_pipeline(agent, sample_jobs)
        ranks = [j["priority_rank"] for j in result]
        assert ranks == [1, 2, 3]


class TestReportFormat:
    """Validate the report() output from run()."""

    def test_run_returns_report_dict(self, agent, sample_jobs):
        report = agent.run(sample_jobs)
        assert isinstance(report, dict)
        assert "agent" in report
        assert "total_jobs" in report
        assert "priority_distribution" in report

    def test_report_job_count_matches(self, agent, sample_jobs):
        report = agent.run(sample_jobs)
        assert report["total_jobs"] == 3

    def test_report_has_distribution(self, agent, sample_jobs):
        report = agent.run(sample_jobs)
        dist = report["priority_distribution"]
        total = sum(dist.values())
        assert total == 3


class TestHeapqExtraction:
    """Validate heapq-based priority queue helpers."""

    def test_pop_highest_returns_top(self, agent, sample_jobs):
        _run_pipeline(agent, sample_jobs)
        top = agent.pop_highest()
        assert top is not None
        assert top["segment_id"] == "SEG-HIGH"
        assert len(agent.get_priority_queue()) == 2

    def test_peek_does_not_remove(self, agent, sample_jobs):
        _run_pipeline(agent, sample_jobs)
        top = agent.peek_highest()
        assert top is not None
        assert len(agent.get_priority_queue()) == 3

    def test_drain_top_n(self, agent, sample_jobs):
        _run_pipeline(agent, sample_jobs)
        top_2 = agent.drain_top_n(2)
        assert len(top_2) == 2
        assert len(agent.get_priority_queue()) == 1

    def test_pop_on_empty_returns_none(self, agent):
        assert agent.pop_highest() is None

    def test_peek_on_empty_returns_none(self, agent):
        assert agent.peek_highest() is None


class TestEdgeCases:
    """Edge-case handling."""

    def test_empty_input_report(self, agent):
        report = agent.run([])
        assert isinstance(report, dict)
        assert report["total_jobs"] == 0

    def test_single_job(self, agent):
        result = _run_pipeline(agent, [{
            "segment_id": "SEG-ONLY",
            "severity_score": 0.5,
            "defect_types": [],
            "traffic": "medium",
            "risk_score": 0.3,
            "estimated_cost": 5000,
        }])
        assert len(result) == 1
        assert result[0]["priority_rank"] == 1

    def test_dict_input_wrapped(self, agent):
        """A single dict (not list) should still work."""
        result = _run_pipeline(agent, {
            "segment_id": "SEG-DICT",
            "severity_score": 0.6,
            "defect_types": ["D20_Alligator_Crack"],
            "traffic": "high",
            "risk_score": 0.5,
            "estimated_cost": 8000,
        })
        assert len(result) == 1
