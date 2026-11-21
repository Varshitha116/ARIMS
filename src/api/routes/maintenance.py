#!/usr/bin/env python3
"""
Maintenance & Scheduling API routes for ARIMS.

Provides endpoints for querying and managing maintenance schedules,
budget status, and road segment health — backed by the in-memory
agent state (and optionally the MySQL database).
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class RepairJobRequest(BaseModel):
    """Manual repair-job submission."""
    segment_id: str = Field(..., description="Road segment identifier")
    severity_score: float = Field(0.5, ge=0.0, le=1.0)
    defect_types: List[str] = Field(default_factory=lambda: ["D40_Pothole"])
    traffic: str = Field("medium", description="low | medium | high | very_high")
    risk_score: float = Field(0.3, ge=0.0, le=1.0)
    estimated_cost: float = Field(5000.0, ge=0.0)
    is_highway: bool = Field(False)


class BudgetUpdateRequest(BaseModel):
    """Update budget parameters."""
    monthly_budget: Optional[float] = None
    daily_capacity: Optional[int] = None
    available_crews: Optional[int] = None


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/maintenance/priority-queue")
async def get_priority_queue():
    """
    Get the current ranked priority queue from the Priority Agent.
    """
    try:
        from agents.priority_agent import PriorityAgent
        agent = PriorityAgent()
        queue = agent.get_priority_queue()
        return {
            "total_jobs": len(queue),
            "priority_queue": queue[:50],  # Limit response size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/prioritize")
async def prioritize_jobs(jobs: List[RepairJobRequest]):
    """
    Submit a batch of repair jobs for MCDA prioritization.

    Returns the jobs ranked by composite score with priority levels assigned.
    """
    from agents.priority_agent import PriorityAgent

    agent = PriorityAgent()
    job_dicts = [j.model_dump() for j in jobs]
    result = agent.run(job_dicts)
    return {
        "total_jobs": result.get("total_jobs", len(job_dicts)),
        "prioritized_jobs": result,
    }


@router.get("/maintenance/schedule")
async def get_maintenance_schedule():
    """
    Get current maintenance schedule from the Scheduler Agent.
    """
    try:
        from agents.scheduler_agent import SchedulerAgent
        agent = SchedulerAgent()
        schedule = agent.get_schedule()
        budget = agent.get_budget_status()
        return {
            "schedule": schedule,
            "budget": budget,
            "total_jobs": len(schedule),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/budget")
async def update_budget(req: BudgetUpdateRequest):
    """
    Update scheduling budget parameters.
    """
    from agents.scheduler_agent import SchedulerAgent
    agent = SchedulerAgent()
    if req.monthly_budget is not None:
        agent.monthly_budget = req.monthly_budget
    if req.daily_capacity is not None:
        agent.daily_capacity = req.daily_capacity
    if req.available_crews is not None:
        agent.available_crews = req.available_crews

    return {
        "updated": True,
        "budget_status": agent.get_budget_status(),
    }


@router.get("/maintenance/db-health")
async def database_health():
    """
    Check whether the database is reachable (optional component).
    """
    try:
        from database.session import check_db_connection
        return check_db_connection()
    except ImportError:
        return {"connected": False, "error": "Database module not available"}


# ============================================================
# BUDGET AGENT ENDPOINTS
# ============================================================

class BudgetAllocateRequest(BaseModel):
    """Request for budget allocation across repair jobs."""
    total_budget: float = Field(100000.0, description="Total budget in INR")
    jobs: List[RepairJobRequest] = Field(..., description="Jobs to allocate budget for")


@router.post("/maintenance/budget-allocate")
async def allocate_budget(req: BudgetAllocateRequest):
    """
    Run Budget Agent: allocate funds across emergency, planned, and
    preventive maintenance tiers using greedy knapsack.
    """
    from agents.budget_agent import BudgetAgent

    agent = BudgetAgent(total_budget=req.total_budget)
    job_dicts = [j.model_dump() for j in req.jobs]
    result = agent.run(job_dicts)
    return result


# ============================================================
# WEATHER AGENT ENDPOINTS
# ============================================================

class WeatherCheckRequest(BaseModel):
    """Request for weather advisory on scheduled jobs."""
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12). Defaults to current month.")
    region: str = Field("inland", description="Region: inland | coastal | northeast | western_ghats | arid")
    jobs: List[RepairJobRequest] = Field(..., description="Scheduled jobs to check")


@router.post("/maintenance/weather-check")
async def weather_check(req: WeatherCheckRequest):
    """
    Run Weather Agent: assess weather risk for scheduled maintenance jobs
    based on Indian monsoon calendar and regional factors.
    """
    from agents.weather_agent import WeatherAgent

    agent = WeatherAgent()
    job_dicts = [j.model_dump() for j in req.jobs]

    # Inject month/region into each job for weather assessment
    for job in job_dicts:
        if req.month is not None:
            job["scheduled_month"] = req.month
        job["region"] = req.region

    result = agent.run(job_dicts)
    return result
