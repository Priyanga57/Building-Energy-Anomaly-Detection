import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Building Energy Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Load Data (Sample for Deployment)
# ==================================================
DATA_PATH = "results/anomaly_labeled_data_sample.csv"

df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ==================================================
# Sidebar Controls
# ==================================================
st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("### ⏱ Time Range Filter")
start_date = st.sidebar.date_input(
    "Start Date", df['timestamp'].min().date()
)
end_date = st.sidebar.date_input(
    "End Date", df['timestamp'].max().date()
)

filtered_df = df[
    (df['timestamp'] >= pd.to_datetime(start_date)) &
    (df['timestamp'] <= pd.to_datetime(end_date))
]

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard demonstrates machine learning–based energy anomaly "
    "detection using a representative sample of real building data."
)

# ==================================================
# Main Title
# ==================================================
st.markdown(
    """
    <h1 style='text-align: center;'>🏢 Building Energy Anomaly Detection</h1>
    <h4 style='text-align: center; color: grey;'>
    Industrial ML for Smart Building Operations & Energy Optimization
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# KPI Section
# ==================================================
total_records = len(filtered_df)
total_anomalies = filtered_df['is_anomaly'].sum()
anomaly_rate = (total_anomalies / total_records) * 100 if total_records > 0 else 0

k1, k2, k3 = st.columns(3)
k1.metric("📊 Total Records", total_records)
k2.metric("🚨 Total Anomalies", total_anomalies)
k3.metric("⚠️ Anomaly Rate (%)", f"{anomaly_rate:.2f}")

# ==================================================
# Tabs Layout
# ==================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Time-Series View",
    "📊 Pattern Analysis",
    "💼 Business Insights"
])

# ==================================================
# TAB 1 — Time-Series Plot
# ==================================================
with tab1:
    st.subheader("Energy Consumption with Detected Anomalies")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        filtered_df['timestamp'],
        filtered_df['electricity'],
        label="Normal",
        alpha=0.8
    )

    anomalies = filtered_df[filtered_df['is_anomaly'] == 1]
    ax.scatter(
        anomalies['timestamp'],
        anomalies['electricity'],
        color="red",
        s=20,
        label="Anomaly"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity Consumption")
    ax.legend()

    st.pyplot(fig)

# ==================================================
# TAB 2 — Pattern Analysis
# ==================================================
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⏰ Anomalies by Hour of Day")
        hourly = (
            filtered_df[filtered_df['is_anomaly'] == 1]
            .groupby('hour')
            .size()
        )
        st.bar_chart(hourly)

    with c2:
        st.subheader("🤝 Ensemble Agreement Strength")
        votes = filtered_df['anomaly_votes'].value_counts().sort_index()
        st.bar_chart(votes)

# ==================================================
# TAB 3 — Business Insights
# ==================================================
with tab3:
    st.subheader("Actionable Business Insights")

    st.success("🔴 Night-time anomalies indicate idle energy waste.")
    st.warning("⚡ Spike anomalies suggest equipment inefficiency or faults.")
    st.info("📅 Weekend anomalies highlight scheduling mismatches.")
    st.success("🤝 Multi-model agreement increases confidence in alerts.")

    st.markdown("---")
    st.markdown("### 🎯 Recommended Actions")
    st.markdown("""
    - Optimize HVAC and lighting schedules  
    - Prioritize maintenance during high-risk periods  
    - Monitor anomaly trends for preventive maintenance  
    - Use anomaly insights to reduce operational costs  
    """)

# ==================================================
# Footer
# ==================================================
st.divider()
st.caption(
    "🚀 Deployed with Streamlit | End-to-End Energy Anomaly Detection Project"
)
