#!/usr/bin/env python3
"""
Weather Agent for ARIMS Multi-Agent System

Adjusts maintenance scheduling based on weather conditions and seasonal
patterns. Particularly important for Indian road infrastructure where
monsoon season (June-September) severely limits repair operations.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from agents.base_agent import BaseAgent


class WeatherAgent(BaseAgent):
    """
    Agent responsible for weather-aware scheduling adjustments.

    Indian Monsoon Calendar:
        June:      Early monsoon
        July-Aug:  Peak monsoon (only ~30% workable days)
        September: Late monsoon
        Oct-Nov:   Post-monsoon (best repair window)
        Dec-Feb:   Winter
        Mar-May:   Summer (hot, but workable)

    Recommendations:
        PROCEED:                  weather_risk < 0.3
        DELAY_1_WEEK:             0.3 <= weather_risk < 0.6
        DELAY_TO_POST_MONSOON:    weather_risk >= 0.6, non-emergency
        PROCEED_WITH_PROTECTION:  weather_risk >= 0.6, P1 emergency
    """

    MONSOON_MONTHS = {6: "early", 7: "peak", 8: "peak", 9: "late"}

    SEASON_WORKABILITY = {
        "monsoon_peak": 0.30,       # Only 30% of days are workable
        "monsoon_early": 0.50,
        "monsoon_late": 0.50,
        "post_monsoon": 0.95,       # Best repair window
        "winter": 0.90,
        "summer": 0.85,
    }

    REGION_FACTORS = {
        "coastal": 1.3,             # Cyclone risk
        "northern_plains": 1.0,     # Baseline
        "deccan_plateau": 0.9,      # Moderate rainfall
        "northeast": 1.4,           # Highest rainfall in India
        "western_ghats": 1.5,       # Very heavy monsoon
        "arid": 0.7,                # Rajasthan/Gujarat — low rain
        "default": 1.0,
    }

    def __init__(
        self,
        agent_id: str = "weather_agent_01",
        region: str = "default",
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "WeatherAgent", config)
        self.region = region
        self._recommendations: List[Dict] = []

    # ── Season helpers ────────────────────────────────────────

    @staticmethod
    def get_current_season(month: int) -> str:
        """Map a calendar month to the Indian road-work season."""
        if month in (7, 8):
            return "monsoon_peak"
        if month == 6:
            return "monsoon_early"
        if month == 9:
            return "monsoon_late"
        if month in (10, 11):
            return "post_monsoon"
        if month in (12, 1, 2):
            return "winter"
        return "summer"  # 3, 4, 5

    def get_workable_days(self, month: int, region: Optional[str] = None) -> int:
        """Estimate workable days in a given month and region."""
        season = self.get_current_season(month)
        workability = self.SEASON_WORKABILITY.get(season, 0.85)
        region_factor = self.REGION_FACTORS.get(
            region or self.region, 1.0
        )
        # Adjust: higher region factor = fewer workable days
        effective_workability = max(0.1, workability / region_factor)
        return max(1, int(30 * effective_workability))

    # ── Pipeline ──────────────────────────────────────────────

    def perceive(self, input_data: Any) -> Any:
        """
        Receive scheduled jobs and weather context.

        Args:
            input_data: dict with:
                - jobs: list of scheduled repair job dicts
                - month: int (1-12, default=current month)
                - region: str (optional override)
        """
        if isinstance(input_data, dict):
            jobs = input_data.get("jobs", [])
            month = input_data.get("month", datetime.now().month)
            region = input_data.get("region", self.region)
        elif isinstance(input_data, list):
            jobs = input_data
            month = datetime.now().month
            region = self.region
        else:
            jobs = []
            month = datetime.now().month
            region = self.region

        season = self.get_current_season(month)
        self._log(
            "Perceive",
            f"Received {len(jobs)} jobs | month={month} | season={season} | region={region}"
        )

        return {
            "jobs": jobs,
            "month": month,
            "region": region,
            "season": season,
        }

    def analyze(self, observations: Any) -> Any:
        """Evaluate weather risk for each scheduled job."""
        jobs = observations["jobs"]
        month = observations["month"]
        region = observations["region"]
        season = observations["season"]

        workability = self.SEASON_WORKABILITY.get(season, 0.85)
        region_mult = self.REGION_FACTORS.get(region, 1.0)

        analyzed_jobs = []
        for job in jobs:
            # Base weather risk: inverse of workability, scaled by region
            base_risk = (1.0 - workability) * region_mult

            # Longer jobs are more weather-sensitive
            duration = job.get("duration_hours", 4)
            duration_factor = min(1.5, 1.0 + (duration - 4) * 0.05)

            weather_risk = min(1.0, base_risk * duration_factor)

            analyzed_jobs.append({
                **job,
                "weather_risk": round(weather_risk, 4),
                "season": season,
                "workable_days": self.get_workable_days(month, region),
            })

        self._log("Analyze", f"Weather risk computed for {len(analyzed_jobs)} jobs")
        return {
            "jobs": analyzed_jobs,
            "season": season,
            "region": region,
            "month": month,
        }

    def decide(self, analysis: Any) -> Any:
        """Generate weather-based scheduling recommendations."""
        jobs = analysis["jobs"]
        recommendations = []

        for job in jobs:
            risk = job["weather_risk"]
            priority = job.get("priority_level", "P3_MEDIUM")
            is_emergency = priority == "P1_EMERGENCY"

            if risk < 0.3:
                recommendation = "PROCEED"
                reason = "Low weather risk — safe to proceed"
            elif risk < 0.6:
                recommendation = "DELAY_1_WEEK"
                reason = "Moderate weather risk — delay by 1 week recommended"
            elif is_emergency:
                recommendation = "PROCEED_WITH_PROTECTION"
                reason = "High weather risk but P1 emergency — proceed with weather protection measures"
            else:
                recommendation = "DELAY_TO_POST_MONSOON"
                reason = "High weather risk — defer to post-monsoon window (Oct-Nov)"

            recommendations.append({
                **job,
                "weather_recommendation": recommendation,
                "weather_reason": reason,
            })

        proceed = sum(1 for r in recommendations if r["weather_recommendation"] == "PROCEED")
        delayed = sum(1 for r in recommendations if "DELAY" in r["weather_recommendation"])
        protected = sum(1 for r in recommendations if r["weather_recommendation"] == "PROCEED_WITH_PROTECTION")

        self._log(
            "Decide",
            f"Proceed: {proceed}, Delayed: {delayed}, Protected: {protected}"
        )

        return {
            "recommendations": recommendations,
            "summary": {
                "proceed": proceed,
                "delay_1_week": sum(1 for r in recommendations if r["weather_recommendation"] == "DELAY_1_WEEK"),
                "delay_post_monsoon": sum(1 for r in recommendations if r["weather_recommendation"] == "DELAY_TO_POST_MONSOON"),
                "proceed_with_protection": protected,
            },
            "season": analysis["season"],
            "region": analysis["region"],
        }

    def execute(self, plan: Any) -> Any:
        """Store recommendations and notify scheduler."""
        self._recommendations = plan["recommendations"]

        self.send_message(
            receiver="scheduler_agent_01",
            msg_type="weather_advisory",
            payload={
                "season": plan["season"],
                "region": plan["region"],
                "summary": plan["summary"],
                "delayed_jobs": [
                    {
                        "segment_id": r["segment_id"],
                        "weather_recommendation": r["weather_recommendation"],
                        "weather_risk": r["weather_risk"],
                    }
                    for r in plan["recommendations"]
                    if "DELAY" in r["weather_recommendation"]
                ],
            },
            priority=2,
        )

        return plan

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate weather assessment report."""
        return {
            "agent": self.agent_id,
            "season": results["season"],
            "region": results["region"],
            "total_jobs": len(results["recommendations"]),
            "summary": results["summary"],
            "weather_risk_avg": round(
                sum(r["weather_risk"] for r in results["recommendations"])
                / max(1, len(results["recommendations"])),
                4,
            ),
        }

    # ── Helpers ───────────────────────────────────────────────

    def get_weather_recommendations(self) -> List[Dict]:
        """Get current weather recommendations."""
        return self._recommendations

