#!/usr/bin/env python3
"""
ARIMS — SQLAlchemy ORM Models

Relational schema for persistent road infrastructure data.
All tables use MySQL-compatible types and follow these conventions:
  - snake_case table names
  - UUID-style string primary keys (for portability)
  - created_at / updated_at timestamps on every table
  - Indexes on frequently queried columns (severity, priority, status)
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean,
    ForeignKey, Index, Enum as SAEnum, JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ARIMS ORM models."""
    pass


# ============================================================
# ROADS
# ============================================================

class Road(Base):
    """A road or road segment being monitored."""

    __tablename__ = "roads"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    segment_id = Column(String(50), unique=True, nullable=False, index=True)
    length_km = Column(Float, default=1.0)
    material = Column(String(50), default="asphalt")
    age_years = Column(Float, default=0.0)
    current_pci = Column(Float, default=100.0)
    current_condition = Column(String(30), default="Good")
    traffic_level = Column(String(30), default="medium")
    climate = Column(String(30), default="temperate")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_highway = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    defects = relationship("RoadDefect", back_populates="road", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="road", cascade="all, delete-orphan")
    maintenance_records = relationship("MaintenanceRecord", back_populates="road", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_roads_condition", "current_condition"),
        Index("ix_roads_pci", "current_pci"),
    )


# ============================================================
# ROAD DEFECTS
# ============================================================

class RoadDefect(Base):
    """An individual defect detected on a road segment."""

    __tablename__ = "road_defects"

    id = Column(String(36), primary_key=True)
    road_id = Column(String(36), ForeignKey("roads.id"), nullable=False, index=True)
    detection_id = Column(String(36), ForeignKey("detections.id"), nullable=True, index=True)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    area_pixels = Column(Float, default=0.0)
    area_ratio = Column(Float, default=0.0)
    severity = Column(String(20), nullable=False, index=True)
    severity_score = Column(Float, default=0.0)
    repair_urgency = Column(String(100))
    estimated_cost_inr = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    road = relationship("Road", back_populates="defects")

    __table_args__ = (
        Index("ix_defects_severity_score", "severity_score"),
    )


# ============================================================
# DETECTIONS (one row per image analysis run)
# ============================================================

class Detection(Base):
    """Record of a single detection run (one image analysed)."""

    __tablename__ = "detections"

    id = Column(String(36), primary_key=True)
    road_id = Column(String(36), ForeignKey("roads.id"), nullable=True, index=True)
    image_path = Column(String(500))
    image_width = Column(Integer)
    image_height = Column(Integer)
    model_name = Column(String(100))
    num_detections = Column(Integer, default=0)
    overall_severity = Column(String(20), index=True)
    overall_severity_score = Column(Float, default=0.0)
    inference_time_ms = Column(Float, default=0.0)
    total_time_ms = Column(Float, default=0.0)
    defect_summary = Column(JSON, nullable=True)
    detection_date = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    road = relationship("Road", back_populates="detections")

    __table_args__ = (
        Index("ix_detections_date_severity", "detection_date", "overall_severity"),
    )


# ============================================================
# MAINTENANCE RECORDS (completed work)
# ============================================================

class MaintenanceRecord(Base):
    """Record of completed or ongoing maintenance work."""

    __tablename__ = "maintenance_records"

    id = Column(String(36), primary_key=True)
    road_id = Column(String(36), ForeignKey("roads.id"), nullable=False, index=True)
    maintenance_type = Column(String(100))
    priority_level = Column(String(30), index=True)
    priority_score = Column(Float, default=0.0)
    status = Column(String(30), default="PENDING", index=True)
    assigned_crew = Column(String(50))
    estimated_cost = Column(Float, default=0.0)
    actual_cost = Column(Float, nullable=True)
    duration_hours = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    scheduled_date = Column(DateTime, nullable=True)
    completed_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    road = relationship("Road", back_populates="maintenance_records")


# ============================================================
# MAINTENANCE SCHEDULE (planned work queue)
# ============================================================

class MaintenanceSchedule(Base):
    """A planned maintenance schedule entry."""

    __tablename__ = "maintenance_schedule"

    id = Column(String(36), primary_key=True)
    segment_id = Column(String(50), nullable=False, index=True)
    priority_level = Column(String(30), index=True)
    composite_score = Column(Float, default=0.0)
    estimated_cost = Column(Float, default=0.0)
    scheduled_date = Column(DateTime, nullable=True)
    scheduled_day = Column(Integer, default=0)
    assigned_crew = Column(String(50))
    schedule_status = Column(String(30), default="PENDING", index=True)
    duration_hours = Column(Float, default=0.0)
    budget_period = Column(String(20))       # e.g. "2026-10"
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_schedule_status_date", "schedule_status", "scheduled_date"),
    )
