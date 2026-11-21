#!/usr/bin/env python3
"""
FastAPI Backend for ARIMS

Main application entry point with CORS, error handling, structured logging,
and auto-generated OpenAPI docs.
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes.detection import router as detection_router
from src.api.routes.agents import router as agents_router
from src.api.routes.simulator import router as simulator_router
from src.api.routes.maintenance import router as maintenance_router


# ============================================================
# LIFESPAN — optional DB init on startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise optional resources at startup, tear down on shutdown."""
    try:
        from database.session import init_db
        init_db()
    except Exception:
        pass  # Database is optional — ARIMS works without it
    yield


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="ARIMS API",
    description=(
        "Agentic AI-Based Autonomous Road Infrastructure Maintenance System. "
        "Provides endpoints for defect detection, multi-agent scheduling, "
        "and infrastructure degradation simulation."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ERROR HANDLING
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# ============================================================
# ROUTES
# ============================================================

app.include_router(detection_router, prefix="/api", tags=["Detection"])
app.include_router(agents_router, prefix="/api", tags=["Agents"])
app.include_router(simulator_router, prefix="/api", tags=["Simulator"])
app.include_router(maintenance_router, prefix="/api", tags=["Maintenance"])

# Serve browser-native dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ARIMS API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


@app.get("/dashboard", tags=["Dashboard"])
async def serve_dashboard():
    """Serve the browser-native ARIMS dashboard."""
    dashboard_path = Path(__file__).parent / "static" / "index.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    return JSONResponse({"error": "Dashboard not found"}, status_code=404)


@app.get("/api/datasets", tags=["Datasets"])
async def list_datasets():
    """List available road defect datasets."""
    try:
        from datasets.dataset_config import DatasetManager
        mgr = DatasetManager()
        mgr.add_dataset("rdd2022_india")
        mgr.add_dataset("rdd2022_us")
        mgr.add_dataset("egypt_roads")
        return mgr.get_combined_stats()
    except Exception as e:
        return {"error": str(e)}
