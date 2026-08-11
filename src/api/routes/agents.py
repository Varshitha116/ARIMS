#!/usr/bin/env python3
"""Agent management API routes for ARIMS."""

import sys
import tempfile
import shutil
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import AgentOrchestrator

router = APIRouter()

_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


@router.get("/agents/status")
async def get_agents_status():
    """Get status of all agents in the multi-agent system."""
    orch = get_orchestrator()
    return orch.get_system_status()


@router.post("/agents/run-pipeline")
async def run_full_pipeline(file: UploadFile = File(...)):
    """
    Run the full multi-agent pipeline on an uploaded image.

    Pipeline: Detection → Degradation → Priority → Schedule → Monitor
    """
    suffix = Path(file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        orch = get_orchestrator()
        result = orch.run_full_pipeline(tmp_path)
        return result
    finally:
        os.unlink(tmp_path)


@router.get("/agents/logs")
async def get_agent_logs():
    """Get logs from all agents."""
    orch = get_orchestrator()
    return orch.get_all_agent_logs()


@router.get("/schedule")
async def get_schedule():
    """Get current repair schedule."""
    orch = get_orchestrator()
    schedule = orch.scheduler_agent.get_schedule()
    budget = orch.scheduler_agent.get_budget_status()
    return {
        "schedule": [
            {
                "segment_id": j.get("segment_id", ""),
                "priority_level": j.get("priority_level", ""),
                "scheduled_date": j.get("scheduled_date", "N/A"),
                "assigned_crew": j.get("assigned_crew", "N/A"),
                "status": j.get("schedule_status", ""),
                "estimated_cost": j.get("estimated_cost", 0),
            }
            for j in schedule
        ],
        "budget": budget,
    }
