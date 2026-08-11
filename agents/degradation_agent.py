#!/usr/bin/env python3
"""
Degradation Prediction Agent for ARIMS

Predicts future road condition (PCI - Pavement Condition Index) based on
current defects, historical data, weather impact, and traffic patterns.
Uses exponential decay model with environmental multipliers.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent


# ============================================================
# DEGRADATION MODELS
# ============================================================

class DegradationModel:
    """
    Road segment degradation prediction using exponential decay.

    PCI(t) = PCI_0 * exp(-λ * t) + environmental_impact

    Where:
        PCI_0: Initial Pavement Condition Index (0-100)
        λ: Decay rate (depends on material, traffic, climate)
        t: Time in years
    """

    # Base decay rates by road material
    MATERIAL_DECAY = {
        "asphalt": 0.06,
        "concrete": 0.03,
        "gravel": 0.12,
        "composite": 0.05,
    }

    # Weather impact multipliers
    WEATHER_MULTIPLIERS = {
        "tropical": 1.4,       # High heat + moisture
        "temperate": 1.0,      # Baseline
        "arid": 0.8,           # Low moisture
        "continental": 1.3,    # Freeze-thaw cycles
        "polar": 1.5,          # Extreme cold + ice
    }

    # Traffic impact multipliers (vehicles/day)
    TRAFFIC_MULTIPLIERS = {
        "low": 0.8,            # < 1000 vpd
        "medium": 1.0,         # 1000-5000 vpd
        "high": 1.3,           # 5000-20000 vpd
        "very_high": 1.6,      # > 20000 vpd
    }

    @staticmethod
    def predict_pci(
        initial_pci: float,
        years: float,
        material: str = "asphalt",
        climate: str = "temperate",
        traffic: str = "medium",
        defect_density: float = 0.0,
    ) -> float:
        """
        Predict PCI at a future time.

        Args:
            initial_pci: Current PCI (0-100)
            years: Number of years into the future
            material: Road surface material
            climate: Climate zone
            traffic: Traffic volume category
            defect_density: Current defects per km (0-50+)

        Returns:
            Predicted PCI (0-100)
        """
        base_decay = DegradationModel.MATERIAL_DECAY.get(material, 0.06)
        weather_mult = DegradationModel.WEATHER_MULTIPLIERS.get(climate, 1.0)
        traffic_mult = DegradationModel.TRAFFIC_MULTIPLIERS.get(traffic, 1.0)

        # Defect density accelerates decay
        defect_mult = 1.0 + (defect_density * 0.02)

        # Combined decay rate
        decay_rate = base_decay * weather_mult * traffic_mult * defect_mult

        # Exponential decay
        predicted_pci = initial_pci * math.exp(-decay_rate * years)

        # Add noise for realism (±2%)
        noise = random.gauss(0, 1.0)
        predicted_pci = max(0, min(100, predicted_pci + noise))

        return round(predicted_pci, 2)

    @staticmethod
    def predict_trajectory(
        initial_pci: float,
        years: int,
        interval_months: int = 3,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate a PCI trajectory over time.

        Returns list of {time_years, pci, condition} points.
        """
        trajectory = []
        steps = int(years * 12 / interval_months)

        for step in range(steps + 1):
            t = step * interval_months / 12.0
            pci = DegradationModel.predict_pci(initial_pci, t, **kwargs)

            # Classify condition
            if pci >= 85:
                condition = "Excellent"
            elif pci >= 70:
                condition = "Good"
            elif pci >= 55:
                condition = "Fair"
            elif pci >= 40:
                condition = "Poor"
            else:
                condition = "Very Poor"

            trajectory.append({
                "time_years": round(t, 2),
                "pci": pci,
                "condition": condition,
            })

        return trajectory

    @staticmethod
    def estimate_remaining_life(current_pci: float, threshold: float = 40.0, **kwargs) -> float:
        """
        Estimate remaining service life (years until PCI drops below threshold).

        Uses binary search to find the crossing point.
        """
        if current_pci <= threshold:
            return 0.0

        low, high = 0.0, 50.0
        for _ in range(50):  # Binary search iterations
            mid = (low + high) / 2.0
            predicted = DegradationModel.predict_pci(current_pci, mid, **kwargs)
            if predicted > threshold:
                low = mid
            else:
                high = mid

        return round((low + high) / 2.0, 1)


# ============================================================
# DEGRADATION AGENT
# ============================================================

class DegradationAgent(BaseAgent):
    """
    Agent that predicts infrastructure degradation for road segments.

    Pipeline:
        perceive: Collect current road segment data
        analyze:  Run degradation models
        decide:   Identify at-risk segments
        execute:  Generate risk scores and timelines
        report:   Publish degradation forecasts
    """

    def __init__(
        self,
        agent_id: str = "degradation_agent_01",
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "DegradationAgent", config)
        self.model = DegradationModel()
        self._prediction_cache: Dict[str, Any] = {}

    def perceive(self, input_data: Any) -> Any:
        """
        Collect road segment data.

        Args:
            input_data: dict or list of dicts with segment information:
                - segment_id: str
                - current_pci: float (0-100)
                - material: str
                - climate: str
                - traffic: str
                - defect_density: float
                - age_years: float
        """
        if isinstance(input_data, dict):
            segments = [input_data]
        elif isinstance(input_data, list):
            segments = input_data
        else:
            segments = []

        self._log("Perceive", f"Received {len(segments)} road segment(s)")
        return segments

    def analyze(self, observations: Any) -> Any:
        """Run degradation prediction for each segment."""
        segments = observations
        predictions = []

        for seg in segments:
            seg_id = seg.get("segment_id", "unknown")
            current_pci = seg.get("current_pci", 75.0)
            material = seg.get("material", "asphalt")
            climate = seg.get("climate", "temperate")
            traffic = seg.get("traffic", "medium")
            defect_density = seg.get("defect_density", 0.0)

            # Generate 10-year trajectory
            trajectory = DegradationModel.predict_trajectory(
                initial_pci=current_pci,
                years=10,
                interval_months=6,
                material=material,
                climate=climate,
                traffic=traffic,
                defect_density=defect_density,
            )

            # Estimate remaining life
            remaining_life = DegradationModel.estimate_remaining_life(
                current_pci=current_pci,
                threshold=40.0,
                material=material,
                climate=climate,
                traffic=traffic,
                defect_density=defect_density,
            )

            # 1-year, 3-year, 5-year forecasts
            pci_1yr = DegradationModel.predict_pci(current_pci, 1.0,
                                                    material=material, climate=climate,
                                                    traffic=traffic, defect_density=defect_density)
            pci_3yr = DegradationModel.predict_pci(current_pci, 3.0,
                                                    material=material, climate=climate,
                                                    traffic=traffic, defect_density=defect_density)
            pci_5yr = DegradationModel.predict_pci(current_pci, 5.0,
                                                    material=material, climate=climate,
                                                    traffic=traffic, defect_density=defect_density)

            prediction = {
                "segment_id": seg_id,
                "current_pci": current_pci,
                "material": material,
                "climate": climate,
                "traffic": traffic,
                "defect_density": defect_density,
                "pci_1yr": pci_1yr,
                "pci_3yr": pci_3yr,
                "pci_5yr": pci_5yr,
                "remaining_life_years": remaining_life,
                "trajectory": trajectory,
            }
            predictions.append(prediction)

        self._log("Analyze", f"Generated predictions for {len(predictions)} segments")
        return predictions

    def decide(self, analysis: Any) -> Any:
        """Identify at-risk segments and assign risk scores."""
        predictions = analysis
        risk_assessments = []

        for pred in predictions:
            # Risk score: combination of current condition and degradation rate
            current_risk = max(0, (100 - pred["current_pci"]) / 100)
            future_risk = max(0, (100 - pred["pci_3yr"]) / 100)
            life_risk = max(0, 1.0 - (pred["remaining_life_years"] / 10.0))

            risk_score = 0.3 * current_risk + 0.4 * future_risk + 0.3 * life_risk
            risk_score = min(1.0, risk_score)

            # Risk category
            if risk_score >= 0.7:
                risk_category = "CRITICAL"
                action = "IMMEDIATE_INTERVENTION"
            elif risk_score >= 0.5:
                risk_category = "HIGH"
                action = "PLAN_REPAIR_6_MONTHS"
            elif risk_score >= 0.3:
                risk_category = "MODERATE"
                action = "SCHEDULE_MAINTENANCE_1_YEAR"
            else:
                risk_category = "LOW"
                action = "ROUTINE_MONITORING"

            assessment = {
                **pred,
                "risk_score": round(risk_score, 4),
                "risk_category": risk_category,
                "recommended_action": action,
            }
            risk_assessments.append(assessment)

        # Sort by risk score (highest first)
        risk_assessments.sort(key=lambda x: x["risk_score"], reverse=True)
        self._log("Decide", f"Risk assessments: {len(risk_assessments)} segments evaluated")
        return risk_assessments

    def execute(self, plan: Any) -> Any:
        """Store predictions and publish to message bus."""
        assessments = plan

        for assessment in assessments:
            seg_id = assessment["segment_id"]
            self._prediction_cache[seg_id] = assessment

            # Publish high-risk segments
            if assessment["risk_score"] >= 0.5:
                self.send_message(
                    receiver="",
                    msg_type="degradation_alert",
                    payload={
                        "segment_id": seg_id,
                        "risk_score": assessment["risk_score"],
                        "risk_category": assessment["risk_category"],
                        "recommended_action": assessment["recommended_action"],
                        "remaining_life_years": assessment["remaining_life_years"],
                    },
                    priority=1 if assessment["risk_score"] >= 0.7 else 2,
                )

        return assessments

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate degradation prediction report."""
        assessments = results
        critical = sum(1 for a in assessments if a["risk_category"] == "CRITICAL")
        high = sum(1 for a in assessments if a["risk_category"] == "HIGH")

        return {
            "agent": self.agent_id,
            "segments_analyzed": len(assessments),
            "critical_segments": critical,
            "high_risk_segments": high,
            "assessments": [
                {
                    "segment_id": a["segment_id"],
                    "current_pci": a["current_pci"],
                    "pci_1yr": a["pci_1yr"],
                    "pci_3yr": a["pci_3yr"],
                    "pci_5yr": a["pci_5yr"],
                    "risk_score": a["risk_score"],
                    "risk_category": a["risk_category"],
                    "remaining_life_years": a["remaining_life_years"],
                    "recommended_action": a["recommended_action"],
                }
                for a in assessments
            ],
        }

    def get_cached_prediction(self, segment_id: str) -> Optional[Dict]:
        """Get cached prediction for a segment."""
        return self._prediction_cache.get(segment_id)
