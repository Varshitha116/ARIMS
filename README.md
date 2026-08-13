# ARIMS: Agentic AI-Based Autonomous Road Infrastructure Maintenance System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers 4.40+](https://img.shields.io/badge/transformers-4.40%2B-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous municipal infrastructure maintenance platform driven by specialized AI agents that collaboratively detect road surface defects using vision transformers, forecast degradation via Monte Carlo simulation, score repair priorities using MCDA, and optimize municipal repair schedules under strict budget constraints.

---

## 🚀 Key Features

1. **Transformer Defect Detector**: Fine-tuned **DETR (DEtection TRansformer)** model for road damage classification (`D00_Longitudinal_Crack`, `D10_Transverse_Crack`, `D20_Alligator_Crack`, `D40_Pothole`) with bounding box, confidence, and severity scoring.
2. **Multi-Agent Framework**: 5 specialized agents (**Detection, Degradation, Priority, Scheduling, Monitoring**) orchestrated by an event-driven `MessageBus` pipeline.
3. **Predictive Degradation Simulator**: Monte Carlo simulation engine implementing state-based Markov Chains and continuous PCI decay models based on traffic, climate, and pavement material.
4. **Municipal Optimization Dashboard**: Interactive 5-page Streamlit web dashboard for real-time defect analysis, agent monitoring, pavement life forecasting, and budget scheduling (denominated in INR ₹).
5. **Real-World Multi-Country Datasets**: Complete integration pipeline supporting **RDD2022 (US subset)**, **Road Surface Dataset (Egypt)**, and **GAPS (German Asphalt Pavement Scanner)** specifications.

---

## 🛠️ Prerequisites & Requirements

- **Python**: 3.10 or higher (3.10, 3.11, 3.12, 3.14 supported)
- **OS**: macOS, Linux, or Windows
- **Hardware**: CPU (Apple Silicon / x86_64) or GPU (NVIDIA CUDA / Apple MPS)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk Space**: ~3-5 GB for datasets and model checkpoints

---

## 📥 Installation & Environment Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/Varshitha116/ARIMS.git
cd ARIMS
```

### Step 2: Create & Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

ARIMS utilizes real-world road damage datasets organized under the `datasets/` directory:

```
datasets/
├── rdd2022/         # RDD2022 Real Road Defect Dataset (600 images, 1,381 bboxes)
│   ├── train/       # 420 train images + YOLO labels
│   ├── val/         # 90 val images + YOLO labels
│   ├── test/        # 90 test images + YOLO labels
│   └── rdd2022.yaml # Dataset YAML config
├── road_surface/    # Real Road Surface Distress Dataset (3,563 images)
└── gaps/            # GAPS technical specifications & access documentation
```

### Reproducible Dataset Pipeline Command
To prepare or re-split the real dataset:
```bash
python scripts/prepare_real_dataset.py
```

---

## 🏋️ Training Transformer & Baseline Models

### 1. Fine-Tune DETR Transformer Detector
```bash
python training/train_rtdetr.py --epochs 10 --batch 2 --device cpu
```

### 2. Fine-Tune YOLOv8 Baseline Detector
```bash
python training/train_yolo.py --data datasets/rdd2022/rdd2022.yaml --epochs 5 --batch 16 --device cpu
```

Model checkpoints are automatically saved to `models/checkpoints/`.

---

## 🔬 Benchmark Evaluation

Run genuine, un-fabricated model evaluation on the real test set images:
```bash
python evaluation/benchmark.py
```

**Output Files:**
- `docs/evaluation_report.md`: Complete markdown evaluation report
- `evaluation/benchmark_results.json`: Metric dictionary (Precision, Recall, F1, mAP@0.5, Latency, FPS)

---

## 🛣️ End-to-End System Pipeline Validation

Validate all 5 transitions of the multi-agent pipeline (`Observation → Detection → Degradation → Priority → Scheduling → Monitoring`):
```bash
python scripts/test_end2end.py
```

---

## 📊 Running the Dashboard & REST API

### 1. Municipal Streamlit Dashboard
```bash
streamlit run main.py
```
Open browser at `http://localhost:8501`.

**Dashboard Pages:**
1. **🔍 Detect Road Damage**: Upload road photo → view bboxes, defect classes, severity, cost estimates.
2. **🤖 AI Agent Control Panel**: Monitor real-time status of Detection, Degradation, Priority, Scheduler, Monitoring agents.
3. **📈 Road Life Predictor**: Run Monte Carlo degradation forecasts for 1-10 year horizons.
4. **📅 Municipal Maintenance Planner**: View prioritized repair schedule and budget allocation in INR (₹).
5. **📊 Performance Reports**: Model comparison and evaluation metrics.

### 2. FastAPI REST Backend
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
API Documentation: `http://localhost:8000/docs`

---

## 📊 Genuine Experimental Results

Evaluated directly on 90 real RDD2022 test images with 221 ground truth defect annotations (CPU Apple M2):

| Model | mAP@0.5 | Precision | Recall | F1 | Latency | FPS |
|-------|---------|-----------|--------|----|---------|-----|
| **YOLOv8 Baseline (conf=0.25)** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **51.8 ms** | **19.3** |
| **YOLOv8 Validation Set (conf=0.25)** | **0.1858** | **0.4510** | **0.2258** | **0.3009** | **51.8 ms** | **19.3** |
| **DETR Transformer Detector (conf=0.05)** | **0.0058** | **0.0109** | **0.4434** | **0.0213** | **401.8 ms** | **2.5** |

---

## 🔧 Troubleshooting

- **Missing PyTorch / torchvision**: Re-install via `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu`.
- **Streamlit Port Busy**: Run `streamlit run main.py --server.port 8502`.
- **OpenCV Window Error**: Ensure `opencv-python` is installed (`pip install opencv-python`).

---

## 📝 License
This project is released under the [MIT License](LICENSE).