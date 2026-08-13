#!/usr/bin/env python3
"""
ARIMS Municipal Repair Optimization Dashboard

Premium multi-page Streamlit dashboard with 5 views:
1. Road Defect Detection (Real AI inference)
2. Multi-Agent System Monitor
3. Degradation Simulator
4. Repair Schedule Optimizer
5. Analytics & Evaluation

Run: streamlit run main.py
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import cv2

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import RoadDefectDetector
from agents.orchestrator import AgentOrchestrator
from agents.degradation_agent import DegradationModel
from simulator.degradation_simulator import (
    DegradationSimulator, RoadSegment, generate_sample_road_network,
    CONDITION_STATES, MAINTENANCE_COSTS
)
from evaluation.metrics import generate_comparison_table


# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="ARIMS - Road Infrastructure AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }
    .metric-card h3 {
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: #e0e7ff;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Severity badges */
    .severity-critical { color: #ef4444; font-weight: 700; }
    .severity-high { color: #f97316; font-weight: 600; }
    .severity-medium { color: #eab308; font-weight: 500; }
    .severity-low { color: #22c55e; font-weight: 500; }

    /* Agent status indicators */
    .agent-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .status-idle { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .status-running { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
    .status-error { background: rgba(239, 68, 68, 0.15); color: #ef4444; }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .main-header h1 {
        color: #e0e7ff;
        font-size: 1.8rem;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        margin: 4px 0 0 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED INITIALIZATION
# ============================================================

@st.cache_resource
def init_detector(model_type: str = "detr"):
    """Initialize the defect detector (cached)."""
    return RoadDefectDetector(model_type=model_type)


@st.cache_resource
def init_orchestrator(model_type: str = "detr"):
    """Initialize the agent orchestrator (cached)."""
    return AgentOrchestrator(model_type=model_type)


@st.cache_resource
def init_simulator():
    """Initialize the degradation simulator (cached)."""
    return DegradationSimulator(seed=42)


@st.cache_resource
def init_road_network():
    """Generate sample road network (cached)."""
    return generate_sample_road_network(20)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🛣️ ARIMS")
    st.markdown("**Autonomous Road Infrastructure**")
    st.markdown("**Maintenance System**")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🔍 Detect Road Damage",
            "🤖 AI Agent Control Panel",
            "📉 Road Life Predictor",
            "📅 Maintenance Planner",
            "📊 Performance Reports",
        ],
        index=0,
    )

    st.divider()
    st.caption("v1.0.0 | ARIMS Project")
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ============================================================
# PAGE 1: DEFECT DETECTION
# ============================================================

if page == "🔍 Detect Road Damage":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Detect Road Damage</h1>
        <p>Upload a road image and let AI identify cracks, potholes & surface defects</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Road Image")
        uploaded_file = st.file_uploader(
            "Upload a road image for analysis",
            type=["jpg", "png", "jpeg"],
            help="Supported: JPEG, PNG images of road surfaces"
        )

        model_choice = st.selectbox(
            "Select Object Detection Model",
            ["DETR Transformer Detector (Ours)", "YOLOv8 Baseline"],
            index=0,
            help="Choose the fine-tuned DETR Transformer model or YOLOv8 baseline"
        )
        selected_model_type = "detr" if "DETR" in model_choice else "yolov8"

        default_conf = 0.05 if selected_model_type == "detr" else 0.25
        confidence = st.slider(
            "Detection Confidence Threshold",
            0.01, 0.90, default_conf, 0.01,
            help="Lower = more detections, Higher = more confident"
        )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        with col1:
            st.image(image, caption="Original Image", width="stretch")

        # Run detection
        with st.spinner(f"🔄 Running {model_choice} defect detection..."):
            detector = init_detector(selected_model_type)
            detector.confidence_threshold = confidence
            result = detector.detect(img_array)

        with col2:
            st.subheader("🎯 Detection Results")

            # Draw annotated image
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            annotated = detector.draw_detections(img_bgr, result.detections)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Defects", width="stretch")

        # Metrics row
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Defects Found", len(result.detections))
        m2.metric("Overall Severity", result.overall_severity)
        m3.metric("Severity Score", f"{result.overall_severity_score:.2f}")
        m4.metric("Inference Time", f"{result.inference_time_ms:.1f} ms")
        m5.metric("Total Time", f"{result.total_time_ms:.1f} ms")

        # Defect details table
        if result.detections:
            st.subheader("📋 Defect Details")

            df_data = []
            for i, det in enumerate(result.detections, 1):
                df_data.append({
                    "#": i,
                    "Type": det.class_name.replace("_", " "),
                    "Confidence": f"{det.confidence:.1%}",
                    "Severity": det.severity,
                    "Score": f"{det.severity_score:.2f}",
                    "Area (px)": f"{det.area_pixels:.0f}",
                    "Urgency": det.repair_urgency,
                    "Est. Cost (₹)": f"₹{det.estimated_cost_inr:,.0f}",
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df, width="stretch", hide_index=True)

            # Defect distribution chart
            if result.defect_summary:
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = px.pie(
                        names=list(result.defect_summary.keys()),
                        values=list(result.defect_summary.values()),
                        title="Defect Type Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hole=0.4,
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                    )
                    st.plotly_chart(fig, width="stretch")

                with col_b:
                    severity_counts = {}
                    for det in result.detections:
                        severity_counts[det.severity] = severity_counts.get(det.severity, 0) + 1

                    fig = px.bar(
                        x=list(severity_counts.keys()),
                        y=list(severity_counts.values()),
                        title="Severity Distribution",
                        color=list(severity_counts.keys()),
                        color_discrete_map={
                            "CRITICAL": "#ef4444", "HIGH": "#f97316",
                            "MEDIUM": "#eab308", "LOW": "#22c55e",
                        },
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, width="stretch")

            # Agentic AI Decision
            st.subheader("🤖 Agentic AI Decision")
            if result.overall_severity == "CRITICAL":
                st.error("🚨 **Emergency Agent**: Immediate repair team dispatched. Road closure recommended.")
            elif result.overall_severity == "HIGH":
                st.warning("⚠️ **Traffic Agent**: High-priority repair scheduled within 48 hours.")
            elif result.overall_severity == "MEDIUM":
                st.info("📋 **Scheduling Agent**: Repair queued for next maintenance window.")
            else:
                st.success("✅ **Monitoring Agent**: Road condition acceptable. Under observation.")
        else:
            st.success("✅ No defects detected. Road surface appears to be in good condition.")


# ============================================================
# PAGE 2: MULTI-AGENT MONITOR
# ============================================================

elif page == "🤖 AI Agent Control Panel":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Agent Control Panel</h1>
        <p>Monitor the autonomous AI agents that detect, prioritize & schedule road repairs</p>
    </div>
    """, unsafe_allow_html=True)

    orch = init_orchestrator()

    # Agent status cards
    st.subheader("Agent Status Panel")

    agents = [
        ("🔍 Detection Agent", orch.detection_agent),
        ("📉 Degradation Agent", orch.degradation_agent),
        ("⚡ Priority Agent", orch.priority_agent),
        ("📅 Scheduler Agent", orch.scheduler_agent),
        ("📊 Monitoring Agent", orch.monitoring_agent),
    ]

    cols = st.columns(5)
    for col, (name, agent) in zip(cols, agents):
        status = agent.get_status()
        state = status["state"]
        state_color = {
            "IDLE": "🟢", "PERCEIVING": "🔵", "ANALYZING": "🔵",
            "DECIDING": "🟡", "EXECUTING": "🟠", "REPORTING": "🟣",
            "ERROR": "🔴", "TERMINATED": "⚫",
        }.get(state, "⚪")

        with col:
            st.markdown(f"**{name}**")
            st.markdown(f"{state_color} {state}")
            st.metric("Runs", status["run_count"])
            st.caption(f"Errors: {status['error_count']}")

    st.divider()

    # Agent Architecture Diagram
    st.subheader("🏗️ Agent Architecture")
    st.markdown("""
    ```mermaid
    graph LR
        A[📷 Image Input] --> B[🔍 Detection Agent]
        B --> C[📉 Degradation Agent]
        C --> D[⚡ Priority Agent]
        D --> E[📅 Scheduler Agent]
        E --> F[🚧 Repair Dispatch]
        G[📊 Monitoring Agent] -.-> B
        G -.-> C
        G -.-> D
        G -.-> E
    ```
    """)

    # Run pipeline button
    st.divider()
    st.subheader("🚀 Run Full Pipeline")
    pipeline_image = st.file_uploader(
        "Upload image to run through all agents", type=["jpg", "png", "jpeg"],
        key="pipeline_upload"
    )

    if pipeline_image and st.button("▶️ Execute Full Pipeline", type="primary"):
        with st.spinner("Running multi-agent pipeline..."):
            image = Image.open(pipeline_image)
            img_array = np.array(image)
            result = orch.run_full_pipeline(img_array)

        st.success(f"✅ Pipeline complete in {result.get('total_pipeline_ms', 0):.0f}ms")

        # Show stage results
        for stage_name, stage_result in result.get("stages", {}).items():
            with st.expander(f"📦 {stage_name.title()} Stage", expanded=False):
                st.json(stage_result)

    # Agent message log
    st.divider()
    st.subheader("📨 Message Bus Log")
    from agents.base_agent import get_message_bus
    msg_history = get_message_bus().get_history(limit=20)
    if msg_history:
        df_msgs = pd.DataFrame(msg_history)
        st.dataframe(df_msgs, width="stretch", hide_index=True)
    else:
        st.info("No messages yet. Run a pipeline to see agent communication.")


# ============================================================
# PAGE 3: DEGRADATION SIMULATOR
# ============================================================

elif page == "📉 Road Life Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>📉 Road Life Predictor</h1>
        <p>Simulate how road condition changes over time — with and without maintenance</p>
    </div>
    """, unsafe_allow_html=True)

    simulator = init_simulator()

    st.info("""
    ℹ️ **Measured Data vs. Simulation Assumptions**:
    - **Measured Input (Real Data)**: Current Pavement Condition Index (PCI), defect density, and crack severity mapped directly from AI defect detections.
    - **Simulation Model (Physics/Markov Assumptions)**: Exponential pavement decay rates ($k_{\text{asphalt}}=0.06/\text{yr}$), environmental climate multipliers, traffic wear coefficients, and stochastic Markov transition matrices.
    """)

    # Simulation controls
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_pci = st.slider("Initial PCI", 20, 100, 75, help="Pavement Condition Index (0-100)")
        material = st.selectbox("Material", ["asphalt", "concrete", "composite", "gravel"])
    with col2:
        climate = st.selectbox("Climate Zone", ["temperate", "continental", "tropical", "arid", "polar"])
        traffic = st.selectbox("Traffic Level", ["low", "medium", "high", "very_high"])
    with col3:
        sim_years = st.slider("Simulation Years", 1, 20, 10)
        num_sims = st.slider("Monte Carlo Runs", 50, 500, 100, 50)

    if st.button("🔬 Run Simulation", type="primary"):
        segment = RoadSegment(
            segment_id="SIM-001",
            current_pci=float(initial_pci),
            material=material,
            climate=climate,
            traffic_level=traffic,
        )
        segment.current_condition = simulator.pci_to_condition(segment.current_pci)

        with st.spinner("Running Monte Carlo simulation..."):
            results = simulator.run_comparison(segment, years=sim_years, num_simulations=num_sims)

        no_maint = results["no_maintenance"]
        with_maint = results["with_maintenance"]

        # PCI Trajectory Comparison Chart
        st.subheader("📈 PCI Trajectory Comparison")

        fig = go.Figure()

        # No maintenance trajectory
        years_nm = [p["year"] for p in no_maint.mean_trajectory]
        pci_nm = [p["pci"] for p in no_maint.mean_trajectory]
        p10_nm = [p["pci"] for p in no_maint.p10_trajectory]
        p90_nm = [p["pci"] for p in no_maint.p90_trajectory]

        fig.add_trace(go.Scatter(
            x=years_nm, y=p90_nm, mode="lines", line=dict(width=0),
            showlegend=False, name="No Maint P90"
        ))
        fig.add_trace(go.Scatter(
            x=years_nm, y=p10_nm, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(239, 68, 68, 0.15)",
            showlegend=False, name="No Maint P10"
        ))
        fig.add_trace(go.Scatter(
            x=years_nm, y=pci_nm, mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            name="No Maintenance (Mean)"
        ))

        # With maintenance trajectory
        years_wm = [p["year"] for p in with_maint.mean_trajectory]
        pci_wm = [p["pci"] for p in with_maint.mean_trajectory]
        p10_wm = [p["pci"] for p in with_maint.p10_trajectory]
        p90_wm = [p["pci"] for p in with_maint.p90_trajectory]

        fig.add_trace(go.Scatter(
            x=years_wm, y=p90_wm, mode="lines", line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=years_wm, y=p10_wm, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(34, 197, 94, 0.15)",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=years_wm, y=pci_wm, mode="lines+markers",
            line=dict(color="#22c55e", width=3),
            name="With Maintenance (Mean)"
        ))

        # Condition zones
        fig.add_hrect(y0=85, y1=100, fillcolor="rgba(34,197,94,0.08)", line_width=0,
                      annotation_text="Excellent", annotation_position="right")
        fig.add_hrect(y0=70, y1=85, fillcolor="rgba(132,204,22,0.08)", line_width=0,
                      annotation_text="Good", annotation_position="right")
        fig.add_hrect(y0=55, y1=70, fillcolor="rgba(234,179,8,0.08)", line_width=0,
                      annotation_text="Fair", annotation_position="right")
        fig.add_hrect(y0=40, y1=55, fillcolor="rgba(249,115,22,0.08)", line_width=0,
                      annotation_text="Poor", annotation_position="right")
        fig.add_hrect(y0=0, y1=40, fillcolor="rgba(239,68,68,0.08)", line_width=0,
                      annotation_text="Very Poor", annotation_position="right")

        fig.update_layout(
            xaxis_title="Years",
            yaxis_title="Pavement Condition Index (PCI)",
            yaxis_range=[0, 105],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=500,
            legend=dict(x=0.01, y=0.01),
        )
        st.plotly_chart(fig, width="stretch")

        # Comparison metrics
        st.subheader("📊 Scenario Comparison")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final PCI (No Maint)", f"{no_maint.final_pci_mean:.1f}",
                   delta=f"{no_maint.final_pci_mean - initial_pci:.1f}")
        c2.metric("Final PCI (With Maint)", f"{with_maint.final_pci_mean:.1f}",
                   delta=f"{with_maint.final_pci_mean - initial_pci:.1f}")
        c3.metric("Avg. Maint Cost/Year", f"₹{with_maint.total_maintenance_cost / sim_years:,.0f}")
        c4.metric("PCI Saved by Maint", f"+{with_maint.final_pci_mean - no_maint.final_pci_mean:.1f}")

        # Final condition distribution
        col_a, col_b = st.columns(2)
        with col_a:
            fig_nm = px.pie(
                names=list(no_maint.final_condition_distribution.keys()),
                values=list(no_maint.final_condition_distribution.values()),
                title="Final Condition (No Maintenance)",
                color=list(no_maint.final_condition_distribution.keys()),
                color_discrete_map={s: CONDITION_STATES[s]["color"] for s in CONDITION_STATES},
                hole=0.4,
            )
            fig_nm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_nm, width="stretch")

        with col_b:
            fig_wm = px.pie(
                names=list(with_maint.final_condition_distribution.keys()),
                values=list(with_maint.final_condition_distribution.values()),
                title="Final Condition (With Maintenance)",
                color=list(with_maint.final_condition_distribution.keys()),
                color_discrete_map={s: CONDITION_STATES[s]["color"] for s in CONDITION_STATES},
                hole=0.4,
            )
            fig_wm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_wm, width="stretch")


# ============================================================
# PAGE 4: REPAIR SCHEDULER
# ============================================================

elif page == "📅 Maintenance Planner":
    st.markdown("""
    <div class="main-header">
        <h1>📅 Maintenance Planner</h1>
        <p>AI-driven repair prioritization, crew assignment & budget optimization</p>
    </div>
    """, unsafe_allow_html=True)

    # Generate sample repair jobs
    road_network = init_road_network()

    st.subheader("🗺️ Road Network Overview")
    df_network = pd.DataFrame([seg.to_dict() for seg in road_network])
    df_display = df_network[["segment_id", "name", "length_km", "material",
                             "age_years", "current_pci", "current_condition",
                             "traffic_level", "defect_count"]].copy()
    df_display.columns = ["ID", "Road Name", "Length (km)", "Material", "Age (yrs)",
                          "PCI", "Condition", "Traffic", "Defects"]

    # Color-code by condition
    st.dataframe(df_display, width="stretch", hide_index=True)

    # PCI Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_network, x="current_pci", nbins=20,
            title="PCI Distribution Across Network",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0")
        st.plotly_chart(fig, width="stretch")

    with col2:
        condition_counts = df_network["current_condition"].value_counts()
        fig = px.pie(
            names=condition_counts.index, values=condition_counts.values,
            title="Condition Distribution",
            color=condition_counts.index,
            color_discrete_map={s: CONDITION_STATES[s]["color"] for s in CONDITION_STATES},
            hole=0.4,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, width="stretch")

    # Run scheduler
    st.divider()
    st.subheader("📋 Generate Repair Schedule")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        budget = st.number_input("Monthly Budget (₹)", 500000, 50000000, 8300000, 500000)
    with col_b:
        capacity = st.number_input("Daily Job Capacity", 1, 10, 3)
    with col_c:
        crews = st.number_input("Available Crews", 1, 20, 5)

    if st.button("🗓️ Generate Schedule", type="primary"):
        orch = init_orchestrator()
        orch.scheduler_agent.monthly_budget = float(budget)
        orch.scheduler_agent.daily_capacity = capacity
        orch.scheduler_agent.available_crews = crews

        # Create repair jobs from road network
        jobs = []
        for seg in road_network:
            if seg.current_pci < 80:  # Only schedule segments needing repair
                severity = max(0, (100 - seg.current_pci) / 100)
                jobs.append({
                    "segment_id": seg.segment_id,
                    "severity_score": severity,
                    "defect_types": ["D20_Alligator_Crack" if seg.current_pci < 50 else "D00_Longitudinal_Crack"],
                    "traffic": seg.traffic_level,
                    "risk_score": severity * 0.8,
                    "estimated_cost": MAINTENANCE_COSTS.get(seg.current_condition, 1000),
                    "is_highway": seg.traffic_level == "very_high",
                })

        with st.spinner("Running priority + scheduling agents..."):
            result = orch.run_scheduling_only(jobs)

        schedule_report = result.get("scheduling", {})
        schedule = schedule_report.get("schedule", [])

        # Budget summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scheduled Jobs", schedule_report.get("scheduled_jobs", 0))
        m2.metric("Deferred Jobs", schedule_report.get("deferred_jobs", 0))
        m3.metric("Budget Used", f"₹{schedule_report.get('budget_used', 0):,.0f}")
        m4.metric("Utilization", f"{schedule_report.get('budget_utilization', 0):.1f}%")

        # Schedule table
        if schedule:
            st.subheader("📅 Repair Schedule")
            df_schedule = pd.DataFrame(schedule)
            display_cols = ["segment_id", "priority_level", "scheduled_date",
                            "assigned_crew", "schedule_status", "estimated_cost",
                            "duration_hours"]
            available_cols = [c for c in display_cols if c in df_schedule.columns]
            st.dataframe(df_schedule[available_cols], width="stretch", hide_index=True)

            # Gantt-style chart
            scheduled_only = [j for j in schedule if j.get("schedule_status") == "SCHEDULED"]
            if scheduled_only:
                gantt_data = []
                for j in scheduled_only[:15]:
                    start = datetime.now() + timedelta(days=j.get("scheduled_day", 0))
                    end = start + timedelta(hours=j.get("duration_hours", 4))
                    gantt_data.append({
                        "Segment": j.get("segment_id", ""),
                        "Start": start,
                        "End": end,
                        "Priority": j.get("priority_level", "P3"),
                        "Crew": j.get("assigned_crew", ""),
                    })

                df_gantt = pd.DataFrame(gantt_data)
                fig = px.timeline(
                    df_gantt, x_start="Start", x_end="End", y="Segment",
                    color="Priority", title="Repair Timeline",
                    color_discrete_map={
                        "P1_EMERGENCY": "#ef4444", "P2_HIGH": "#f97316",
                        "P3_MEDIUM": "#eab308", "P4_LOW": "#22c55e",
                    },
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", height=400,
                )
                st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE 5: ANALYTICS & EVALUATION
# ============================================================

# ============================================================
# PAGE 5: ANALYTICS & EVALUATION
# ============================================================

elif page == "📊 Performance Reports":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Performance Reports</h1>
        <p>Genuine model accuracy, speed benchmarks & real-world evaluation on RDD2022 dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Load genuine evaluation benchmark results
    bench_file = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
    bench_data = {}
    if bench_file.exists():
        try:
            import json
            with open(bench_file, "r") as f:
                bench_data = json.load(f)
        except Exception:
            pass

    st.subheader("🏆 Model Performance Comparison (Real RDD2022 Benchmark)")
    st.markdown("""
    **Genuine evaluation metrics** computed on **90 real RDD2022 test images** containing **221 ground truth defect annotations** (CPU Apple M2):
    """)

    comparison_data = {
        "Model": [
            "DETR Transformer Detector (conf=0.05)",
            "YOLOv8 Baseline (Val Split, conf=0.25)",
            "YOLOv8 Baseline (Test Split, conf=0.25)",
            "EfficientDet (Pham 2023 - Literature)",
            "YOLOv5 (Arya 2022 - Literature)",
            "Faster R-CNN (Zhang 2021 - Literature)",
        ],
        "mAP@0.5": [0.0058, 0.1858, 0.0000, 0.6470, 0.6210, 0.5840],
        "mAP@0.5:0.95": [0.0015, 0.0912, 0.0000, 0.4050, 0.3890, 0.3510],
        "Precision": [0.0109, 0.4510, 0.0000, 0.6980, 0.6720, 0.6450],
        "Recall": [0.4434, 0.2258, 0.0000, 0.6010, 0.5890, 0.5430],
        "F1": [0.0213, 0.3009, 0.0000, 0.6460, 0.6280, 0.5900],
        "Latency (ms)": [401.8, 51.8, 51.8, 68.4, 45.2, 125.0],
        "FPS": [2.5, 19.3, 19.3, 14.6, 22.1, 8.0],
    }

    df_comp = pd.DataFrame(comparison_data)

    def highlight_ours(row):
        if "DETR Transformer" in row["Model"] or "YOLOv8 Baseline" in row["Model"]:
            return ["background-color: rgba(99, 102, 241, 0.15); font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_comp.style.apply(highlight_ours, axis=1).format({
            "mAP@0.5": "{:.4f}", "mAP@0.5:0.95": "{:.4f}",
            "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
            "Latency (ms)": "{:.1f}", "FPS": "{:.1f}",
        }),
        width="stretch", hide_index=True,
    )

    # Performance charts
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_comp, x="Model", y="Recall",
            title="Defect Detection Recall Comparison",
            color="Recall",
            color_continuous_scale="viridis",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_tickangle=-30, height=450,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.scatter(
            df_comp, x="Latency (ms)", y="Recall",
            text="Model", title="Recall vs Latency Speed Trade-off",
            size="FPS", color="Recall",
            color_continuous_scale="turbo",
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=450,
        )
        st.plotly_chart(fig, width="stretch")

    # Per-class performance on real validation split
    st.divider()
    st.subheader("📋 Per-Class Detection Performance (YOLOv8 Real Validation Split)")

    class_data = {
        "Defect Class": [
            "D00 Longitudinal Crack", "D10 Transverse Crack",
            "D20 Alligator Crack",
        ],
        "AP@0.5": [0.350, 0.154, 0.240],
        "Precision": [0.521, 0.410, 0.422],
        "Recall": [0.285, 0.198, 0.194],
        "F1": [0.368, 0.267, 0.266],
        "Ground Truth Boxes": [844, 412, 125],
    }
    df_class = pd.DataFrame(class_data)
    st.dataframe(df_class, width="stretch", hide_index=True)

    # Radar chart
    fig = go.Figure()
    categories = class_data["Defect Class"]
    for metric in ["AP@0.5", "Precision", "Recall"]:
        values = class_data[metric] + [class_data[metric][0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", name=metric,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.6])),
        title="Per-Class Validation Performance Radar",
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")

    # System metrics
    st.divider()
    st.subheader("⚙️ Real-World System Performance Metrics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("DETR Latency (CPU)", "401.8 ms", help="Mean latency per image on Apple M2 CPU")
    s2.metric("YOLOv8 Latency (CPU)", "51.8 ms", help="Mean latency per image on Apple M2 CPU")
    s3.metric("DETR Test Recall", "44.34%", help="Percent of real defect annotations detected")
    s4.metric("Dataset Size", "4,805 images", help="Total real RDD2022 US subset images")