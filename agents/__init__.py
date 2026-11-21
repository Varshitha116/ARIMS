# ARIMS agents package
"""
Multi-agent system for autonomous road infrastructure maintenance.

Agents:
    DetectionAgent      — CV-based road defect detection
    DegradationAgent    — PCI decay prediction
    PriorityAgent       — MCDA scoring + heapq extraction
    BudgetAgent         — Tiered budget allocation
    WeatherAgent        — Monsoon-aware scheduling
    SchedulerAgent      — Constraint-based repair scheduling
    MonitoringAgent     — System health monitoring
    AgentOrchestrator   — Pipeline coordinator
"""
