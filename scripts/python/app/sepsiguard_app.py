import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import streamlit as st
import numpy as np

from inference.predictor import predict_risk


st.set_page_config(page_title="SepsiGuard", layout="wide")


st.title("SepsiGuard")

st.subheader("Early Sepsis Risk Support System")


st.warning("Research prototype. Not a diagnostic device.")


st.divider()


st.header("Patient Trajectory Input")


patient_id = st.text_input("Patient ID", value="Demo Patient")


st.write("Provide the latest 12-hour trajectory.")


# Temporary demo input
# Later replaced with real feature extraction

use_demo = st.checkbox("Use demo trajectory", value=True)


if use_demo:
    sequence = np.random.randn(12, 69).astype(np.float32)


else:
    st.info("Manual trajectory upload will be added.")

    sequence = None


if st.button("Generate Risk Prediction"):
    if sequence is not None:
        risk = predict_risk(sequence)

        st.divider()

        st.header("Prediction")

        st.metric("12-hour Sepsis Risk", f"{risk:.2%}")

        if risk >= 0.5:
            st.error("Elevated predicted risk")

        else:
            st.success("Lower predicted risk")

        st.subheader("Model Information")

        st.write(
            """
            Model:
            Temporal Convolutional Network (TCN)

            Input:
            12-hour clinical trajectory

            Features:
            69 temporal features

            Prediction horizon:
            12 hours
            """
        )
