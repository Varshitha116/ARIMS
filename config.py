#!/usr/bin/env python3
"""
ARIMS — Centralized Configuration

Loads settings from environment variables (or a .env file) so that
secrets, database URLs, and tuning knobs stay outside the source tree.

Usage:
    from config import settings
    print(settings.DATABASE_URL)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Attempt to load .env file at import time (no hard dependency)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars


@dataclass
class Settings:
    """Application-wide settings, populated from environment variables."""

    # ── Project Metadata ──────────────────────────────────────────
    PROJECT_NAME: str = "ARIMS"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Database ──────────────────────────────────────────────────
    # Format: mysql+pymysql://user:password@host:port/dbname
    DATABASE_URL: str = ""
    DB_ECHO: bool = False          # SQLAlchemy echo (SQL logging)
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ── Model / Detection ─────────────────────────────────────────
    MODEL_TYPE: str = "yolov8"     # yolov8 | rtdetr
    MODEL_PATH: str = ""           # Leave empty for auto-discovery
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    DEVICE: str = "cpu"            # cpu | cuda | mps

    # ── API Server ────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: str = "*"     # Comma-separated CORS origins

    # ── Scheduling Defaults ───────────────────────────────────────
    MONTHLY_BUDGET: float = 8_300_000.0   # INR
    DAILY_CAPACITY: int = 3
    AVAILABLE_CREWS: int = 5

    # ── Priority Weights (must sum to 1.0) ────────────────────────
    WEIGHT_SEVERITY: float = 0.30
    WEIGHT_SAFETY: float = 0.25
    WEIGHT_TRAFFIC: float = 0.20
    WEIGHT_DEGRADATION: float = 0.15
    WEIGHT_COST_EFFICIENCY: float = 0.10

    # ── File Paths ────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Override defaults with environment variables where set."""
        for fld in self.__dataclass_fields__:
            env_val = os.environ.get(fld)
            if env_val is not None:
                current = getattr(self, fld)
                # Cast to the same type as the default
                if isinstance(current, bool):
                    setattr(self, fld, env_val.lower() in ("1", "true", "yes"))
                elif isinstance(current, int):
                    setattr(self, fld, int(env_val))
                elif isinstance(current, float):
                    setattr(self, fld, float(env_val))
                else:
                    setattr(self, fld, env_val)

    @property
    def priority_weights(self) -> dict:
        """Return priority weights as a dict for PriorityAgent."""
        return {
            "severity": self.WEIGHT_SEVERITY,
            "safety": self.WEIGHT_SAFETY,
            "traffic": self.WEIGHT_TRAFFIC,
            "degradation": self.WEIGHT_DEGRADATION,
            "cost_efficiency": self.WEIGHT_COST_EFFICIENCY,
        }

    @property
    def db_available(self) -> bool:
        """True if a DATABASE_URL has been configured."""
        return bool(self.DATABASE_URL)


# Singleton — import this everywhere
settings = Settings()
