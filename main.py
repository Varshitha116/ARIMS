import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ==========================================
# PAGE TITLE
# ==========================================

st.title(
    "AI Road Infrastructure Dashboard"
)

st.subheader(
    "Autonomous Road Damage Detection"
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "png", "jpeg"]
)

# ==========================================
# IMAGE ANALYSIS
# ==========================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    # ======================================
    # CONVERT IMAGE
    # ======================================

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # ======================================
    # EDGE DETECTION
    # ======================================

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    # ======================================
    # DAMAGE ESTIMATION
    # ======================================

    edge_pixels = np.sum(edges > 0)

    total_pixels = edges.shape[0] * edges.shape[1]

    damage_ratio = (
        edge_pixels / total_pixels
    ) * 100

    # ======================================
    # AI ANALYSIS
    # ======================================

    st.subheader("AI Analysis")

    st.write(
        f"Damage Score: "
        f"{round(damage_ratio,2)}%"
    )

    # ======================================
    # SEVERITY
    # ======================================

    if damage_ratio > 15:

        severity = "CRITICAL"

        recommendation = (
            "Immediate Repair Required"
        )

    elif damage_ratio > 10:

        severity = "HIGH"

        recommendation = (
            "Repair within 24 Hours"
        )

    elif damage_ratio > 5:

        severity = "MEDIUM"

        recommendation = (
            "Repair within 3 Days"
        )

    else:

        severity = "LOW"

        recommendation = (
            "Monitor Condition"
        )

    st.write(
        f"Severity: {severity}"
    )

    st.write(
        f"Recommendation: "
        f"{recommendation}"
    )

    # ======================================
    # SHOW EDGE DETECTION
    # ======================================

    st.subheader(
        "Detected Road Damage Areas"
    )

    st.image(
        edges,
        caption="Road Damage Detection",
        use_container_width=True
    )

    # ======================================
    # MULTI-AGENT OUTPUT
    # ======================================

    st.subheader(
        "Agentic AI Decision"
    )

    if severity == "CRITICAL":

        st.error(
            "Emergency Agent: "
            "Repair Team Dispatched"
        )

    elif severity == "HIGH":

        st.warning(
            "Traffic Agent: "
            "High Priority Road"
        )

    else:

        st.success(
            "Monitoring Agent: "
            "Road Under Observation"
        )