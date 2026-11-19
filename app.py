import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Screening System")
st.markdown("""
**M.Tech AI & Data Science Project** | *Educational Screening Prototype*  
> **Disclaimer:** This application is an educational machine learning prototype and is **not** a medical diagnostic tool. Please consult a healthcare professional for clinical diagnosis.
""")

@st.cache_resource
def load_artifacts():
    if os.path.exists("models/best_model.pkl") and os.path.exists("models/scaler.pkl"):
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    return None, None

model, scaler = load_artifacts()

if model is None:
    st.warning("⚠️ Model files not found! Please run `python src/train.py` first.")
else:
    st.subheader("Enter Patient Health Metrics")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=300, value=110)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=30, max_value=200, value=72)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=5, max_value=100, value=23)

    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=10, max_value=900, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=26.5, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=0.45, step=0.01)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)

    if st.button("Predict Diabetes Risk", type="primary"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"**Result: Higher Risk of Diabetes (Positive Class)** — Estimated Probability: **{probability:.1%}**")
        else:
            st.success(f"**Result: Lower Risk of Diabetes (Negative Class)** — Estimated Probability: **{probability:.1%}**")

    if os.path.exists("models/model_comparison.csv"):
        st.divider()
        st.subheader("Model Evaluation Comparison (CO-5)")
        comp_df = pd.read_csv("models/model_comparison.csv")
        st.dataframe(comp_df, use_container_width=True)

    if os.path.exists("models/feature_importance.csv"):
        st.divider()
        st.subheader("Key Risk Factors (Feature Importance)")
        feat_df = pd.read_csv("models/feature_importance.csv").set_index("Feature")
        st.bar_chart(feat_df)
    if os.path.exists("models/feature_importance.csv"):
        import matplotlib.pyplot as plt
        st.divider()
        st.subheader("Key Risk Factors (Feature Importance)")
        feat_df = pd.read_csv("models/feature_importance.csv").sort_values(by="Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(feat_df["Feature"], feat_df["Importance"], color="#1f77b4")
        ax.set_xlabel("Relative Importance Score")
        ax.set_xlim(0, feat_df["Importance"].max() * 1.15)
        st.pyplot(fig)    