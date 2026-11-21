#!/usr/bin/env python3
"""
Tests for ARIMS FastAPI endpoints (src/api/app.py).

Uses FastAPI's TestClient for synchronous HTTP testing — no server needed.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.app import app


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    return TestClient(app)


# ============================================================
# Health Endpoints
# ============================================================

class TestHealthEndpoints:
    """Root and health-check endpoints."""

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "ARIMS API"
        assert data["status"] == "running"

    def test_health_returns_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestDetectionEndpoints:
    """Detection API tests."""

    def test_detect_rejects_non_image(self, client):
        resp = client.post(
            "/api/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_detections_list_returns_200(self, client):
        resp = client.get("/api/detections")
        assert resp.status_code == 200


class TestAgentEndpoints:
    """Agent management API tests."""

    def test_agents_status_returns_200(self, client):
        resp = client.get("/api/agents/status")
        assert resp.status_code == 200

    def test_agent_logs_returns_200(self, client):
        resp = client.get("/api/agents/logs")
        assert resp.status_code == 200


class TestSimulatorEndpoints:
    """Simulation API tests."""

    def test_road_network_returns_200(self, client):
        resp = client.get("/api/road-network")
        assert resp.status_code == 200
        data = resp.json()
        assert "segments" in data
        assert "total_segments" in data

    def test_simulate_returns_200(self, client):
        resp = client.post("/api/simulate", json={
            "segment_id": "SEG-TEST",
            "initial_pci": 75.0,
            "material": "asphalt",
            "climate": "tropical",
            "traffic": "high",
            "years": 5,
            "num_simulations": 50,
            "with_maintenance": False,
        })
        assert resp.status_code == 200

    def test_simulate_compare_returns_200(self, client):
        resp = client.post("/api/simulate/compare", json={
            "segment_id": "SEG-CMP",
            "initial_pci": 80.0,
            "years": 5,
            "num_simulations": 50,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "no_maintenance" in data
        assert "with_maintenance" in data


class TestMaintenanceEndpoints:
    """Maintenance & scheduling API tests."""

    def test_priority_queue_returns_200(self, client):
        resp = client.get("/api/maintenance/priority-queue")
        assert resp.status_code == 200

    def test_prioritize_batch(self, client):
        jobs = [
            {
                "segment_id": "SEG-A",
                "severity_score": 0.8,
                "defect_types": ["D40_Pothole"],
                "traffic": "high",
                "risk_score": 0.6,
                "estimated_cost": 10000,
                "is_highway": True,
            },
            {
                "segment_id": "SEG-B",
                "severity_score": 0.3,
                "defect_types": ["D00_Longitudinal_Crack"],
                "traffic": "low",
                "risk_score": 0.2,
                "estimated_cost": 3000,
                "is_highway": False,
            },
        ]
        resp = client.post("/api/maintenance/prioritize", json=jobs)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data

    def test_db_health_returns_200(self, client):
        resp = client.get("/api/maintenance/db-health")
        assert resp.status_code == 200
