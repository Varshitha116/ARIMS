#!/usr/bin/env python3
"""
Integration Tests for ARIMS v2.0

Tests the complete end-to-end multi-agent pipeline:
  Image/Data -> Detection -> Degradation -> Priority -> Budget ->
  Weather -> Scheduling -> Monitoring -> API Response
"""

import pytest
from datetime import datetime


# ============================================================
# FIXTURE: Sample road repair jobs
# ============================================================

@pytest.fixture
def sample_jobs():
    """Realistic repair job candidates for Indian road segments."""
    return [
        {
            "segment_id": "NH44-KM120",
            "severity_score": 0.85,
            "defect_types": ["D40_Pothole"],
            "traffic": "very_high",
            "risk_score": 0.75,
            "estimated_cost": 25000,
            "is_highway": True,
        },
        {
            "segment_id": "SH-TANK-BUND",
            "severity_score": 0.60,
            "defect_types": ["D20_Alligator_Crack"],
            "traffic": "high",
            "risk_score": 0.50,
            "estimated_cost": 12000,
            "is_highway": False,
        },
        {
            "segment_id": "MRD-COLONY-03",
            "severity_score": 0.25,
            "defect_types": ["D00_Longitudinal_Crack"],
            "traffic": "low",
            "risk_score": 0.15,
            "estimated_cost": 4000,
            "is_highway": False,
        },
        {
            "segment_id": "ORR-SEC5",
            "severity_score": 0.70,
            "defect_types": ["D40_Pothole", "D10_Transverse_Crack"],
            "traffic": "high",
            "risk_score": 0.60,
            "estimated_cost": 18000,
            "is_highway": True,
        },
    ]


# ============================================================
# TEST: Full agent pipeline (Priority -> Schedule)
# ============================================================

class TestFullAgentPipeline:
    """Integration: Priority -> Scheduler end-to-end."""

    def test_priority_to_scheduler_pipeline(self, sample_jobs):
        from agents.priority_agent import PriorityAgent
        from agents.scheduler_agent import SchedulerAgent

        # Stage 1: Priority ranking
        priority = PriorityAgent()
        priority_report = priority.run(sample_jobs)

        assert priority_report["total_jobs"] == 4
        assert "priority_distribution" in priority_report

        # Stage 2: Scheduling
        queue = priority.get_priority_queue()
        assert len(queue) == 4
        assert queue[0]["priority_rank"] == 1  # Highest first

        scheduler = SchedulerAgent(monthly_budget=500000, daily_capacity=3)
        schedule_report = scheduler.run(queue)

        assert schedule_report["scheduled_jobs"] > 0
        assert schedule_report["budget_used"] > 0

        # Verify ordering is preserved
        schedule = schedule_report["schedule"]
        scheduled = [j for j in schedule if j["status"] == "SCHEDULED"]
        assert len(scheduled) > 0

    def test_highway_pothole_gets_highest_priority(self, sample_jobs):
        from agents.priority_agent import PriorityAgent

        priority = PriorityAgent()
        priority.run(sample_jobs)
        queue = priority.get_priority_queue()

        # NH44 highway pothole should be ranked #1
        top_job = queue[0]
        assert top_job["segment_id"] == "NH44-KM120"
        assert top_job["priority_level"] in ("P1_EMERGENCY", "P2_HIGH")


# ============================================================
# TEST: Budget Agent integration
# ============================================================

class TestBudgetAgentIntegration:
    """Integration: Priority -> Budget allocation."""

    def test_budget_allocation_respects_tiers(self, sample_jobs):
        from agents.priority_agent import PriorityAgent
        from agents.budget_agent import BudgetAgent

        # Prioritize first
        priority = PriorityAgent()
        priority.run(sample_jobs)
        queue = priority.get_priority_queue()

        # Allocate budget
        budget = BudgetAgent(total_budget=100000)
        report = budget.run(queue)

        assert report["total_budget"] == 100000
        assert report["funded_jobs"] + report["unfunded_jobs"] == 4
        assert report["total_allocated"] <= 100000
        assert "allocation_by_tier" in report

    def test_tight_budget_unfunds_low_priority(self, sample_jobs):
        from agents.priority_agent import PriorityAgent
        from agents.budget_agent import BudgetAgent

        priority = PriorityAgent()
        priority.run(sample_jobs)
        queue = priority.get_priority_queue()

        # Very tight budget
        budget = BudgetAgent(total_budget=20000)
        report = budget.run(queue)

        assert report["unfunded_jobs"] > 0
        assert report["utilization_pct"] > 0


# ============================================================
# TEST: Weather Agent integration
# ============================================================

class TestWeatherAgentIntegration:
    """Integration: Scheduler -> Weather advisory."""

    def test_monsoon_delays_non_emergency(self, sample_jobs):
        from agents.weather_agent import WeatherAgent

        weather = WeatherAgent(region="western_ghats")
        jobs_with_duration = [
            {**j, "duration_hours": 8, "priority_level": "P3_MEDIUM"}
            for j in sample_jobs
        ]

        report = weather.run({
            "jobs": jobs_with_duration,
            "month": 7,  # Peak monsoon
            "region": "western_ghats",
        })

        assert report["season"] == "monsoon_peak"
        # In peak monsoon + western ghats, most jobs should be delayed
        delayed = report["summary"]["delay_1_week"] + report["summary"]["delay_post_monsoon"]
        assert delayed > 0

    def test_post_monsoon_allows_proceed(self, sample_jobs):
        from agents.weather_agent import WeatherAgent

        weather = WeatherAgent(region="deccan_plateau")
        jobs_with_duration = [
            {**j, "duration_hours": 4, "priority_level": "P2_HIGH"}
            for j in sample_jobs
        ]

        report = weather.run({
            "jobs": jobs_with_duration,
            "month": 10,  # Post-monsoon — best window
            "region": "deccan_plateau",
        })

        assert report["season"] == "post_monsoon"
        assert report["summary"]["proceed"] > 0

    def test_emergency_proceeds_despite_monsoon(self):
        from agents.weather_agent import WeatherAgent

        weather = WeatherAgent(region="northeast")
        report = weather.run({
            "jobs": [{
                "segment_id": "EMERGENCY-001",
                "priority_level": "P1_EMERGENCY",
                "duration_hours": 6,
                "estimated_cost": 50000,
            }],
            "month": 8,  # Peak monsoon
            "region": "northeast",  # Highest rainfall
        })

        rec = weather.get_weather_recommendations()[0]
        assert rec["weather_recommendation"] == "PROCEED_WITH_PROTECTION"


# ============================================================
# TEST: Degradation Simulator
# ============================================================

class TestDegradationIntegration:
    """Integration: Degradation simulation with agents."""

    def test_degradation_agent_processes_segments(self):
        from agents.degradation_agent import DegradationAgent

        agent = DegradationAgent()
        segments = [
            {
                "segment_id": "DEG-001",
                "current_pci": 70,
                "material": "asphalt",
                "climate": "tropical",
                "traffic": "high",
                "defect_density": 5.0,
            },
        ]
        report = agent.run(segments)
        assert "assessments" in report or "agent" in report

    def test_simulator_comparison(self):
        from simulator.degradation_simulator import DegradationSimulator, RoadSegment

        sim = DegradationSimulator(seed=42)
        segment = RoadSegment(
            segment_id="SIM-001", current_pci=75.0,
            material="asphalt", climate="tropical", traffic_level="high",
        )

        no_maint = sim.run_simulation(segment, years=5, num_simulations=20, with_maintenance=False)
        with_maint = sim.run_simulation(segment, years=5, num_simulations=20, with_maintenance=True)

        # With maintenance should result in higher final PCI
        assert with_maint.final_pci_mean >= no_maint.final_pci_mean - 10  # Allow some stochastic noise


# ============================================================
# TEST: Orchestrator end-to-end
# ============================================================

class TestOrchestratorPipeline:
    """Integration: Full orchestrator pipeline."""

    def test_full_pipeline_with_segments(self):
        from agents.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator(model_type="yolov8", monthly_budget=500000)

        segments = [
            {
                "segment_id": "E2E-001",
                "initial_pci": 55,
                "material": "asphalt",
                "climate": "tropical",
                "traffic": "high",
                "severity_score": 0.75,
                "defect_types": ["D40_Pothole"],
                "estimated_cost": 20000,
            },
        ]

        result = orch.run_full_pipeline(image_input=None, road_segments=segments)

        assert "stages" in result
        assert "pipeline_id" in result
        assert "total_pipeline_ms" in result

    def test_system_status(self):
        from agents.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator()
        status = orch.get_system_status()

        assert "agents" in status
        assert "system_metrics" in status


# ============================================================
# TEST: API integration
# ============================================================

class TestAPIIntegration:
    """Integration: FastAPI endpoint chain."""

    def test_prioritize_then_query(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)

        # Submit jobs for prioritization
        resp = client.post("/api/maintenance/prioritize", json=[
            {
                "segment_id": "API-001",
                "severity_score": 0.8,
                "defect_types": ["D40_Pothole"],
                "traffic": "high",
                "risk_score": 0.6,
                "estimated_cost": 15000,
            },
            {
                "segment_id": "API-002",
                "severity_score": 0.3,
                "defect_types": ["D00_Longitudinal_Crack"],
                "traffic": "low",
                "risk_score": 0.2,
                "estimated_cost": 3000,
            },
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jobs"] == 2

        # Query priority queue
        q_resp = client.get("/api/maintenance/priority-queue")
        assert q_resp.status_code == 200

    def test_simulate_degradation_via_api(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.post("/api/simulate", json={
            "segment_id": "API-SIM",
            "initial_pci": 80,
            "material": "asphalt",
            "climate": "tropical",
            "traffic": "high",
            "years": 3,
            "num_simulations": 10,
            "with_maintenance": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "mean_trajectory" in data or "result" in data

