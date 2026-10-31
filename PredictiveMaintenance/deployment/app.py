import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configure page styling and title
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Engine Predictive Maintenance System")
st.write("""
This interactive application monitors engine sensor readings in real-time to predict whether a fault condition is imminent.
Use the sidebar input controls to adjust individual sensor measurements.
""")

# Determine absolute directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Securely resolve paths to deployment artifacts
MODEL_PATH = os.path.join(current_dir, "best_model.joblib")
SCALER_PATH = os.path.join(current_dir, "scaler.joblib")

# Verify dependencies
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(f"❌ Required deployment artifacts missing at `{current_dir}`. Please verify repository structure.")
else:
    # Load champion model and normalization scaler
    @st.cache_resource
    def load_artifacts():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    model, scaler = load_artifacts()

    st.sidebar.header("⚙️ Real-time Engine Sensor Telemetry")

    # Generate interactive input controllers matching training parameters
    engine_rpm = st.sidebar.slider("Engine RPM", min_value=100.0, max_value=4000.0, value=2139.0, step=10.0)
    lub_oil_pressure = st.sidebar.slider("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.3, step=0.1)
    fuel_pressure = st.sidebar.slider("Fuel Pressure (bar)", min_value=0.0, max_value=12.0, value=5.0, step=0.1)
    coolant_pressure = st.sidebar.slider("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    lub_oil_temp = st.sidebar.slider("Lub Oil Temperature (°C)", min_value=20.0, max_value=150.0, value=85.0, step=1.0)
    coolant_temp = st.sidebar.slider("Coolant Temperature (°C)", min_value=20.0, max_value=130.0, value=75.0, step=1.0)

    # Organize telemetry into a structured DataFrame
    input_data = pd.DataFrame({
        'Engine rpm': [engine_rpm],
        'Lub oil pressure': [lub_oil_pressure],
        'Fuel pressure': [fuel_pressure],
        'Coolant pressure': [coolant_pressure],
        'lub oil temp': [lub_oil_temp],
        'Coolant temp': [coolant_temp]
    })

    st.subheader("📊 Live Sensor Telemetry")
    st.dataframe(input_data)

    # Transform input space using the scaled training distribution
    input_scaled = scaler.transform(input_data)

    # Execute machine learning inference
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None

    st.subheader("🔮 Diagnostic Assessment")
    if prediction == 1:
        st.error("⚠️ WARNING: Engine Fault Condition Detected!")
        st.write("Immediate preventive maintenance attention is highly recommended to mitigate failure risks.")
    else:
        st.success("✅ NORMAL: Engine Operating Within Safe Thresholds")
        st.write("System parameters indicate healthy operational performance.")

    # Display prediction confidence when available
    if prediction_proba is not None:
        st.write("---")
        st.write(f"**Fault Probability:** {prediction_proba[1] * 100:.2f}%")
        st.write(f"**Normal Probability:** {prediction_proba[0] * 100:.2f}%")
