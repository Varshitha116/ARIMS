#!/usr/bin/env python3
"""
Multi-Agent Base Class for ARIMS

Provides the abstract foundation for all agents in the ARIMS multi-agent
maintenance scheduling framework. Implements state machine, event-driven
communication, and logging.

Agent States:
    IDLE → PERCEIVING → ANALYZING → DECIDING → EXECUTING → REPORTING → IDLE

Communication:
    Agents communicate via a shared message bus (in-process queue).
"""

import time
import uuid
import json
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from collections import deque
from datetime import datetime


# ============================================================
# AGENT STATES & MESSAGES
# ============================================================

class AgentState(Enum):
    IDLE = "IDLE"
    PERCEIVING = "PERCEIVING"
    ANALYZING = "ANALYZING"
    DECIDING = "DECIDING"
    EXECUTING = "EXECUTING"
    REPORTING = "REPORTING"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"


@dataclass
class AgentMessage:
    """Message passed between agents via the message bus."""
    message_id: str = ""
    sender: str = ""
    receiver: str = ""  # Empty = broadcast
    msg_type: str = ""  # e.g., "detection_result", "schedule_request"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    priority: int = 0  # Higher = more urgent

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority,
        }


@dataclass
class AgentLog:
    """Single log entry for agent activity."""
    timestamp: str = ""
    agent_id: str = ""
    state: str = ""
    action: str = ""
    details: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "state": self.state,
            "action": self.action,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


# ============================================================
# MESSAGE BUS (in-process pub/sub)
# ============================================================

class MessageBus:
    """
    Simple in-process message bus for agent communication.

    Agents subscribe to message types. When a message is published,
    all subscribers for that type receive it.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: deque = deque(maxlen=500)
        self._pending: Dict[str, deque] = {}  # agent_id -> message queue

    def subscribe(self, msg_type: str, callback: Callable):
        """Subscribe a callback to a message type."""
        if msg_type not in self._subscribers:
            self._subscribers[msg_type] = []
        self._subscribers[msg_type].append(callback)

    def publish(self, message: AgentMessage):
        """Publish a message to all subscribers."""
        self._history.append(message)

        # Direct message to specific agent
        if message.receiver:
            if message.receiver not in self._pending:
                self._pending[message.receiver] = deque(maxlen=100)
            self._pending[message.receiver].append(message)

        # Broadcast to type subscribers
        for callback in self._subscribers.get(message.msg_type, []):
            try:
                callback(message)
            except Exception as e:
                print(f"⚠️  Message handler error: {e}")

    def get_pending(self, agent_id: str) -> List[AgentMessage]:
        """Get and clear pending messages for an agent."""
        if agent_id in self._pending:
            messages = list(self._pending[agent_id])
            self._pending[agent_id].clear()
            return messages
        return []

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent message history."""
        recent = list(self._history)[-limit:]
        return [m.to_dict() for m in recent]


# Singleton message bus
_global_bus = MessageBus()


def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    return _global_bus


# ============================================================
# BASE AGENT
# ============================================================

class BaseAgent(ABC):
    """
    Abstract base class for all ARIMS agents.

    Subclasses must implement:
        - perceive(input_data) → extracted features/observations
        - analyze(observations) → analysis results
        - decide(analysis) → action plan
        - execute(plan) → execution results
        - report(results) → report/summary

    The run() method orchestrates the full pipeline.
    """

    def __init__(self, agent_id: str, agent_type: str, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.state = AgentState.IDLE
        self.logs: List[AgentLog] = []
        self.bus = get_message_bus()
        self._created_at = datetime.now().isoformat()
        self._last_active = self._created_at
        self._run_count = 0
        self._error_count = 0

    # --------------------------------------------------------
    # STATE MANAGEMENT
    # --------------------------------------------------------

    def _set_state(self, new_state: AgentState):
        """Transition to a new state with logging."""
        old_state = self.state
        self.state = new_state
        self._last_active = datetime.now().isoformat()
        self._log(f"State: {old_state.value} → {new_state.value}")

    def _log(self, action: str, details: str = "", duration_ms: float = 0.0):
        """Add a log entry."""
        entry = AgentLog(
            timestamp=datetime.now().isoformat(),
            agent_id=self.agent_id,
            state=self.state.value,
            action=action,
            details=details,
            duration_ms=duration_ms,
        )
        self.logs.append(entry)

    # --------------------------------------------------------
    # COMMUNICATION
    # --------------------------------------------------------

    def send_message(self, receiver: str, msg_type: str, payload: Dict,
                     priority: int = 0):
        """Send a message to another agent or broadcast."""
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
        )
        self.bus.publish(msg)
        self._log(f"Sent: {msg_type}", f"→ {receiver or 'broadcast'}")

    def receive_messages(self) -> List[AgentMessage]:
        """Receive pending messages."""
        messages = self.bus.get_pending(self.agent_id)
        if messages:
            self._log(f"Received {len(messages)} messages")
        return messages

    # --------------------------------------------------------
    # ABSTRACT METHODS (implement in subclasses)
    # --------------------------------------------------------

    @abstractmethod
    def perceive(self, input_data: Any) -> Any:
        """Extract observations from raw input."""
        pass

    @abstractmethod
    def analyze(self, observations: Any) -> Any:
        """Analyze observations to produce insights."""
        pass

    @abstractmethod
    def decide(self, analysis: Any) -> Any:
        """Make decisions based on analysis."""
        pass

    @abstractmethod
    def execute(self, plan: Any) -> Any:
        """Execute the decided plan."""
        pass

    @abstractmethod
    def report(self, results: Any) -> Dict[str, Any]:
        """Generate a report from execution results."""
        pass

    # --------------------------------------------------------
    # ORCHESTRATION
    # --------------------------------------------------------

    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Run the complete agent pipeline:
        PERCEIVE → ANALYZE → DECIDE → EXECUTE → REPORT

        Returns: Report dictionary
        """
        self._run_count += 1
        pipeline_start = time.perf_counter()

        try:
            # Step 1: Perceive
            self._set_state(AgentState.PERCEIVING)
            t0 = time.perf_counter()
            observations = self.perceive(input_data)
            self._log("Perceive complete", duration_ms=(time.perf_counter() - t0) * 1000)

            # Step 2: Analyze
            self._set_state(AgentState.ANALYZING)
            t0 = time.perf_counter()
            analysis = self.analyze(observations)
            self._log("Analyze complete", duration_ms=(time.perf_counter() - t0) * 1000)

            # Step 3: Decide
            self._set_state(AgentState.DECIDING)
            t0 = time.perf_counter()
            plan = self.decide(analysis)
            self._log("Decide complete", duration_ms=(time.perf_counter() - t0) * 1000)

            # Step 4: Execute
            self._set_state(AgentState.EXECUTING)
            t0 = time.perf_counter()
            results = self.execute(plan)
            self._log("Execute complete", duration_ms=(time.perf_counter() - t0) * 1000)

            # Step 5: Report
            self._set_state(AgentState.REPORTING)
            t0 = time.perf_counter()
            report = self.report(results)
            self._log("Report complete", duration_ms=(time.perf_counter() - t0) * 1000)

            self._set_state(AgentState.IDLE)

            total_ms = (time.perf_counter() - pipeline_start) * 1000
            report["_meta"] = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "run_number": self._run_count,
                "total_pipeline_ms": round(total_ms, 2),
                "state": self.state.value,
            }

            return report

        except Exception as e:
            self._error_count += 1
            self._set_state(AgentState.ERROR)
            self._log("ERROR", str(e))
            return {
                "error": str(e),
                "_meta": {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "state": "ERROR",
                    "error_count": self._error_count,
                }
            }

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "created_at": self._created_at,
            "last_active": self._last_active,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "log_count": len(self.logs),
        }

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent agent logs."""
        return [log.to_dict() for log in self.logs[-limit:]]
