
import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path
st.set_page_config(
    page_title="MachineGuard",
    page_icon="⚙️",
    layout="wide"
)

MODEL_PATH = Path("models/best_model.pkl")
METADATA_PATH = Path("models/model_metadata.json")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files not found. Run python train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

st.title("⚙️ MachineGuard")

st.subheader("Predictive Maintenance and Machine Failure Detection")

st.write(
    "An ML-powered application that predicts "
    "industrial machine failures using operating parameters."
)

st.divider()

st.sidebar.title("MachineGuard")

st.sidebar.info(
    "Enter the operating parameters of a machine "
    "to estimate its failure risk."
)

st.sidebar.write(
    "Model: " + metadata["model_name"]
)

st.header("Machine Operating Parameters")

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1
    )

with col2:

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        min_value=1000,
        max_value=3000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=3.0,
        max_value=80.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=300,
        value=100
    )

if st.button("Predict Machine Failure", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Feature engineering must match training
    input_data["Temperature_difference"] = (
        input_data["Process temperature [K]"]
        - input_data["Air temperature [K]"]
    )

    input_data["Mechanical_load_proxy"] = (
        input_data["Torque [Nm]"]
        * input_data["Rotational speed [rpm]"]
    )

    # Predict
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Failure Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Predicted Class",
            "Failure" if prediction == 1 else "No Failure"
        )

    st.progress(float(probability))

    if prediction == 1:

        st.error(
            "Potential machine failure detected! "
            "Consider inspecting the equipment."
        )

    else:

        st.success(
            "No failure predicted by the model "
            "for the supplied operating conditions."
        )

    st.caption(
        "This is a model prediction, not a guarantee "
        "of actual machine condition. Follow approved "
        "industrial safety and maintenance procedures."
    )

st.divider()

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{metadata['accuracy'] * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{metadata['precision'] * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{metadata['recall'] * 100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{metadata['f1_score'] * 100:.2f}%"
)

st.caption(
    "Performance metrics are based on the held-out test dataset."
)
