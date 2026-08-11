# ARIMS — Agentic AI-Based Autonomous Road Infrastructure Maintenance System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

ARIMS is an AI-powered autonomous system for road infrastructure maintenance. It detects road defects using computer vision, predicts degradation using Monte Carlo simulation, coordinates maintenance through a multi-agent framework, and provides an interactive municipal dashboard.

## Key Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Road Defect Detection Model (YOLOv8 / RT-DETR) | ✅ Complete |
| 2 | Multi-Agent Maintenance Scheduling Framework | ✅ Complete |
| 3 | Predictive Infrastructure Degradation Simulator | ✅ Complete |
| 4 | Municipal Repair Optimization Dashboard | ✅ Complete |

## Architecture

```
📷 Image Input → 🔍 Detection Agent → 📉 Degradation Agent → ⚡ Priority Agent → 📅 Scheduler Agent → 🚧 Repair Dispatch
                                                    ↑                                        ↑
                                              📊 Monitoring Agent ─────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Detection Model | YOLOv8 / RT-DETR (ultralytics) |
| ML Framework | PyTorch 2.x |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data Processing | NumPy, Pandas, OpenCV |
| Agent Framework | Custom MAS with message bus |
| Dataset | RDD2022 (4 defect classes) |

## Installation

```bash
# Clone
git clone https://github.com/Varshitha116/ARIMS.git
cd ARIMS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Dashboard (Recommended)
```bash
streamlit run main.py
```
Visit `http://localhost:8501` — 5 interactive pages.

### API Server
```bash
uvicorn src.api.app:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for OpenAPI documentation.

### Prepare Dataset
```bash
python scripts/download_rdd2022.py
```

### Train Models
```bash
# YOLOv8
python training/train_yolo.py --data datasets/rdd2022/rdd2022.yaml --epochs 100

# RT-DETR
python training/train_rtdetr.py --data datasets/rdd2022/rdd2022.yaml --epochs 100
```

## Project Structure

```
ARIMS/
├── main.py                         # Streamlit dashboard (5 pages)
├── requirements.txt                # Dependencies
├── models/
│   └── detector.py                 # Unified detection module
├── agents/
│   ├── base_agent.py               # Abstract agent + message bus
│   ├── detection_agent.py          # Defect detection agent
│   ├── degradation_agent.py        # Degradation prediction agent
│   ├── priority_agent.py           # MCDA priority ranking agent
│   ├── scheduler_agent.py          # Constraint-based scheduler
│   ├── monitoring_agent.py         # System health monitor
│   └── orchestrator.py             # Central coordinator
├── simulator/
│   └── degradation_simulator.py    # Monte Carlo simulation engine
├── training/
│   ├── train_yolo.py               # YOLOv8 training script
│   └── train_rtdetr.py             # RT-DETR training script
├── evaluation/
│   └── metrics.py                  # mAP, IoU, P/R, F1, latency
├── src/api/
│   ├── app.py                      # FastAPI application
│   └── routes/
│       ├── detection.py            # Detection endpoints
│       ├── agents.py               # Agent management endpoints
│       └── simulator.py            # Simulation endpoints
├── scripts/
│   ├── download_rdd2022.py         # Dataset preparation
│   ├── ingest.py                   # Data ingestion
│   ├── preprocess.py               # Data preprocessing
│   └── validate_data.py            # Data validation
├── docs/
│   └── evaluation_report.md        # Publication-quality evaluation
├── SDD.md                          # Software Design Document
└── PROJECT_JOURNAL.md              # Development log
```

## Performance

| Model | mAP@0.5 | Precision | Recall | F1 | Latency | FPS |
|-------|---------|-----------|--------|-----|---------|-----|
| **ARIMS YOLOv8-m** | **0.724** | **0.756** | **0.689** | **0.721** | **28.4ms** | **35.2** |
| ARIMS RT-DETR-L | 0.698 | 0.731 | 0.671 | 0.700 | 42.1ms | 23.8 |
| YOLOv5-s (Arya 2022) | 0.621 | 0.672 | 0.589 | 0.628 | 18.7ms | 53.5 |
| Faster R-CNN (Zhang 2021) | 0.584 | 0.645 | 0.543 | 0.590 | 125ms | 8.0 |

## Defect Classes (RDD2022)

| Code | Type | Description |
|------|------|-------------|
| D00 | Longitudinal Crack | Cracks parallel to road direction |
| D10 | Transverse Crack | Cracks perpendicular to road direction |
| D20 | Alligator Crack | Interconnected/fatigue cracking |
| D40 | Pothole | Bowl-shaped depressions |

## Documentation

- [Software Design Document](SDD.md)
- [Project Journal](PROJECT_JOURNAL.md)
- [Evaluation Report](docs/evaluation_report.md)

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*ARIMS Project — Osmania University Internship 2026*