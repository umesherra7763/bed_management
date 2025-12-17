import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time

# --------------------------
# CONFIG
# --------------------------
API_URL = "http://localhost:8000"
THEME_BG = "#F3F4F6"

st.set_page_config(
    layout="wide",
    page_title="NeuroBed OS",
    page_icon="🏥"
)

# --------------------------
# GLOBAL CSS — FIXED COLORS & VISIBILITY
# --------------------------
st.markdown(f"""
<style>

    /* Global background */
    .stApp {{
        background-color: {THEME_BG};
    }}

    /* Global text color */
    html, body, p, span, div, label {{
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    h1, h2, h3, h4, h5 {{
        color: #1E293B !important;
        font-weight: 600 !important;
    }}

    /* KPI Cards */
    .css-card {{
        background-color: white;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #2563EB;
        margin-bottom: 16px;
        color: #1E293B !important;
    }}

    .card-warning {{
        border-left-color: #F59E0B !important;
    }}

    .card-danger {{
        border-left-color: #EF4444 !important;
    }}

    /* Fix labels for forms */
    .stTextInput label,
    .stNumberInput label,
    .stSlider label,
    .stSelectbox label {{
        color: #1E293B !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }}

</style>
""", unsafe_allow_html=True)


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("🏥 NeuroBed OS")
page = st.sidebar.radio("Navigation", ["Dashboard Overview", "New Admission", "Bed Manager", "Simulation"])


# --------------------------
# PAGE 1 — DASHBOARD
# --------------------------
if page == "Dashboard Overview":
    st.title("🧠 NeuroBed Command Center")
    st.write("Real-time LOS prediction + bed management intelligence")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(f"""
    <div class="css-card">
        <h3>92%</h3>
        <p>Current Occupancy</p>
    </div>
    """, unsafe_allow_html=True)

    k2.markdown(f"""
    <div class="css-card">
        <h3>4</h3>
        <p>Beds Available</p>
    </div>
    """, unsafe_allow_html=True)

    k3.markdown(f"""
    <div class="css-card">
        <h3>12</h3>
        <p>Predicted Discharges Today</p>
    </div>
    """, unsafe_allow_html=True)

    k4.markdown(f"""
    <div class="css-card card-warning">
        <h3>High</h3>
        <p>Surge Risk</p>
    </div>
    """, unsafe_allow_html=True)


    # Charts
    chart_row = st.columns([2,1])

    # Occupancy Trend
    with chart_row[0]:
        st.subheader("📈 Occupancy Trend (Last 24 Hours)")
        fig = go.Figure(
            go.Scatter(
                y=np.random.randint(82, 96, 24),
                mode='lines+markers',
                line=dict(color="#2563EB"),
                fill='tozeroy'
            )
        )
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Ward Breakdown
    with chart_row[1]:
        st.subheader("🏨 Ward Breakdown")
        fig2 = go.Figure(
            go.Pie(
                labels=['ICU', 'General', 'Trauma'],
                values=[14, 48, 22],
                hole=0.45
            )
        )
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig2, use_container_width=True)


# --------------------------
# PAGE 2 — NEW ADMISSION
# --------------------------
elif page == "New Admission":

    st.title("🧾 Patient Intake & AI LOS Prediction")

    with st.form("admission_form"):
        col1, col2 = st.columns(2)

        patient_id = col1.text_input("Patient ID", "P-10249")
        age = col2.number_input("Age", 18, 100, 65)

        severity = st.slider("Severity Score", 1, 5, 3)
        sofa = st.number_input("SOFA Score", 0, 20, 5)
        cci = st.number_input("CCI Index", 0, 10, 2)

        submitted = st.form_submit_button("Run Prediction Engine")

    if submitted:
        st.markdown("---")
        st.success("Prediction Complete!")

        # Mock Prediction
        pred_los = 4.5
        confidence = 88

        r1, r2 = st.columns(2)

        # LOS Card
        r1.markdown(f"""
        <div class="css-card">
            <h4 style="color:#475569">Predicted LOS</h4>
            <h1 style="font-size:48px; color:#2563EB;">{pred_los} Days</h1>
            <p>Confidence: {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

        # SHAP-like bar chart
        with r2:
            st.subheader("Feature Contribution")
            st.bar_chart({
                "Severity": severity*2,
                "Age": age*0.4,
                "SOFA": sofa*1.8,
                "CCI": cci
            })


# --------------------------
# PAGE 3 — BED MANAGER (coming soon)
# --------------------------
elif page == "Bed Manager":

    st.title("🛏 Bed Manager")
    st.info("Bed Gantt chart and assignment engine will be added here.")


# --------------------------
# PAGE 4 — SIMULATION
# --------------------------
elif page == "Simulation":

    st.title("📊 Capacity Forecast Simulation")

    st.info("Running Monte Carlo Simulation (100 runs, 30 days)")

    days = list(range(1, 31))
    mean_occ = np.linspace(40, 52, 30) + np.random.normal(0, 0.8, 30)
    upper_risk = mean_occ + 4

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=days, y=mean_occ, name='Expected Occupancy', line=dict(color="#2563EB")))
    fig_sim.add_trace(go.Scatter(x=days, y=upper_risk, name='High-Risk Scenario', line=dict(color="red", dash='dash')))

    fig_sim.add_hline(y=50, annotation_text="Max Capacity", line_color="black")

    fig_sim.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))

    st.plotly_chart(fig_sim, use_container_width=True)
