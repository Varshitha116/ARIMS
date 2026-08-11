#!/usr/bin/env python3
"""
Monitoring Agent for ARIMS Multi-Agent System

Tracks system health, agent performance, pipeline execution,
and provides real-time monitoring data for the dashboard.
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

from agents.base_agent import BaseAgent, get_message_bus


class MonitoringAgent(BaseAgent):
    """
    Agent that monitors the health and performance of all other agents.

    Tracks:
        - Agent states and uptime
        - Pipeline execution times
        - Error rates
        - Resource utilization
        - System-wide KPIs
    """

    def __init__(
        self,
        agent_id: str = "monitoring_agent_01",
        config: Optional[Dict] = None,
    ):
        super().__init__(agent_id, "MonitoringAgent", config)
        self._agent_registry: Dict[str, Dict] = {}
        self._system_metrics: Dict[str, Any] = {
            "total_detections": 0,
            "total_defects_found": 0,
            "total_jobs_scheduled": 0,
            "total_simulations_run": 0,
            "avg_inference_time_ms": 0.0,
            "uptime_start": datetime.now().isoformat(),
            "error_count": 0,
        }
        self._pipeline_history: List[Dict] = []
        self._alerts: List[Dict] = []

    def register_agent(self, agent: BaseAgent):
        """Register an agent for monitoring."""
        self._agent_registry[agent.agent_id] = {
            "agent_type": agent.agent_type,
            "registered_at": datetime.now().isoformat(),
            "agent_ref": agent,
        }

    def perceive(self, input_data: Any) -> Any:
        """Collect system-wide metrics from all registered agents."""
        agent_statuses = []

        for agent_id, info in self._agent_registry.items():
            agent = info["agent_ref"]
            status = agent.get_status()
            agent_statuses.append(status)

        # Collect message bus stats
        bus = get_message_bus()
        msg_history = bus.get_history(limit=100)

        return {
            "agent_statuses": agent_statuses,
            "message_count": len(msg_history),
            "timestamp": datetime.now().isoformat(),
        }

    def analyze(self, observations: Any) -> Any:
        """Analyze system health indicators."""
        statuses = observations["agent_statuses"]

        health = {
            "total_agents": len(statuses),
            "active_agents": sum(1 for s in statuses if s["state"] != "TERMINATED"),
            "error_agents": sum(1 for s in statuses if s["state"] == "ERROR"),
            "idle_agents": sum(1 for s in statuses if s["state"] == "IDLE"),
            "agent_details": statuses,
            "total_errors": sum(s.get("error_count", 0) for s in statuses),
            "total_runs": sum(s.get("run_count", 0) for s in statuses),
        }

        return health

    def decide(self, analysis: Any) -> Any:
        """Generate alerts for anomalies."""
        health = analysis
        alerts = []

        # Check for error agents
        if health["error_agents"] > 0:
            alerts.append({
                "level": "CRITICAL",
                "message": f"{health['error_agents']} agent(s) in ERROR state",
                "timestamp": datetime.now().isoformat(),
            })

        # Check error rate
        if health["total_runs"] > 0:
            error_rate = health["total_errors"] / health["total_runs"]
            if error_rate > 0.1:
                alerts.append({
                    "level": "WARNING",
                    "message": f"High error rate: {error_rate:.1%}",
                    "timestamp": datetime.now().isoformat(),
                })

        self._alerts.extend(alerts)
        return {"health": health, "alerts": alerts}

    def execute(self, plan: Any) -> Any:
        """Update system metrics and store health snapshot."""
        self._pipeline_history.append({
            "timestamp": datetime.now().isoformat(),
            "health": plan["health"],
            "alert_count": len(plan["alerts"]),
        })
        return plan

    def report(self, results: Any) -> Dict[str, Any]:
        """Generate monitoring report."""
        health = results["health"]
        return {
            "agent": self.agent_id,
            "system_health": {
                "total_agents": health["total_agents"],
                "active_agents": health["active_agents"],
                "error_agents": health["error_agents"],
                "total_runs": health["total_runs"],
                "total_errors": health["total_errors"],
            },
            "system_metrics": self._system_metrics,
            "recent_alerts": self._alerts[-10:],
            "agent_details": health["agent_details"],
        }

    def update_metrics(self, key: str, value: Any):
        """Update a system metric."""
        self._system_metrics[key] = value

    def increment_metric(self, key: str, amount: int = 1):
        """Increment a counter metric."""
        self._system_metrics[key] = self._system_metrics.get(key, 0) + amount

    def get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        return self._system_metrics

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def get_all_agent_statuses(self) -> List[Dict]:
        """Get status of all registered agents."""
        statuses = []
        for agent_id, info in self._agent_registry.items():
            agent = info["agent_ref"]
            status = agent.get_status()
            status["registered_at"] = info["registered_at"]
            statuses.append(status)
        return statuses
