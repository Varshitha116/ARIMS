#!/usr/bin/env python3
"""Simulator API routes for ARIMS."""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.degradation_simulator import (
    DegradationSimulator, RoadSegment, generate_sample_road_network
)

router = APIRouter()

_simulator = DegradationSimulator(seed=42)
_road_network = generate_sample_road_network(20)


class SimulationRequest(BaseModel):
    segment_id: str = Field(default="SEG-001")
    initial_pci: float = Field(default=75.0, ge=0, le=100)
    material: str = Field(default="asphalt")
    climate: str = Field(default="temperate")
    traffic: str = Field(default="medium")
    years: int = Field(default=10, ge=1, le=30)
    num_simulations: int = Field(default=100, ge=10, le=1000)
    with_maintenance: bool = Field(default=False)


@router.post("/simulate")
async def run_simulation(req: SimulationRequest):
    """Run a degradation simulation for a road segment."""
    segment = RoadSegment(
        segment_id=req.segment_id,
        current_pci=req.initial_pci,
        material=req.material,
        climate=req.climate,
        traffic_level=req.traffic,
    )
    segment.current_condition = _simulator.pci_to_condition(segment.current_pci)

    result = _simulator.run_simulation(
        segment,
        years=req.years,
        num_simulations=req.num_simulations,
        with_maintenance=req.with_maintenance,
    )
    return result.to_dict()


@router.post("/simulate/compare")
async def compare_scenarios(req: SimulationRequest):
    """Run both maintenance and no-maintenance scenarios for comparison."""
    segment = RoadSegment(
        segment_id=req.segment_id,
        current_pci=req.initial_pci,
        material=req.material,
        climate=req.climate,
        traffic_level=req.traffic,
    )
    segment.current_condition = _simulator.pci_to_condition(segment.current_pci)

    results = _simulator.run_comparison(
        segment,
        years=req.years,
        num_simulations=req.num_simulations,
    )
    return {
        "no_maintenance": results["no_maintenance"].to_dict(),
        "with_maintenance": results["with_maintenance"].to_dict(),
    }


@router.get("/road-network")
async def get_road_network():
    """Get the simulated road network."""
    return {
        "segments": [seg.to_dict() for seg in _road_network],
        "total_segments": len(_road_network),
    }
