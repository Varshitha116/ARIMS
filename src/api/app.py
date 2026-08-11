#!/usr/bin/env python3
"""
FastAPI Backend for ARIMS

Main application entry point with CORS, error handling, structured logging,
and auto-generated OpenAPI docs.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.detection import router as detection_router
from src.api.routes.agents import router as agents_router
from src.api.routes.simulator import router as simulator_router


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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ARIMS API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
