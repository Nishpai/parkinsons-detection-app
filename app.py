import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model with mmap_mode=None
model = joblib.load('parkinsons_xgboost_tuned_pipeline.pkl', mmap_mode=None)

st.set_page_config(page_title="Parkinson's Detection", page_icon="🧠", layout="centered")

st.title("🧠 Parkinson's Disease Detection")
st.markdown("### Voice-based Parkinson's Prediction")
st.write("This tool predicts the likelihood of Parkinson's disease using voice measurements.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Frequency Features")

    MDVP_Fo = st.slider(
        "MDVP:Fo(Hz) - Average vocal pitch", 
        min_value=80.0, max_value=300.0, value=154.23, step=0.1,
        help="Average fundamental frequency of the voice (average pitch)."
    )
    MDVP_Fhi = st.slider(
        "MDVP:Fhi(Hz) - Highest vocal pitch", 
        min_value=100.0, max_value=600.0, value=197.10, step=0.1,
        help="Maximum fundamental frequency reached during speech."
    )
    MDVP_Flo = st.slider(
        "MDVP:Flo(Hz) - Lowest vocal pitch", 
        min_value=60.0, max_value=250.0, value=116.32, step=0.1,
        help="Minimum fundamental frequency reached during speech."
    )

    st.subheader("Jitter Features (Voice Pitch Stability)")

    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", value=0.00622, step=0.0001, format="%.5f",
        help="Variation in pitch between consecutive voice cycles. Higher = less stable voice.")
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", value=0.00004, step=0.00001, format="%.6f",
        help="Absolute jitter value.")
    MDVP_RAP = st.number_input("MDVP:RAP", value=0.00331, step=0.0001, format="%.5f",
        help="Relative Average Perturbation – pitch variation over 3 cycles.")
    MDVP_PPQ = st.number_input("MDVP:PPQ", value=0.00345, step=0.0001, format="%.5f",
        help="Pitch Perturbation Quotient – pitch variation over 5 cycles.")
    Jitter_DDP = st.number_input("Jitter:DDP", value=0.00994, step=0.0001, format="%.5f",
        help="Derived from RAP. Another measure of pitch instability.")

with col2:
    st.subheader("Shimmer Features (Voice Loudness Stability)")

    MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.02971, step=0.001, format="%.5f",
        help="Variation in loudness between consecutive voice cycles.")
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.282, step=0.01, format="%.3f",
        help="Shimmer measured in decibels.")
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.01587, step=0.001, format="%.5f",
        help="Amplitude Perturbation Quotient over 3 cycles.")
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.01809, step=0.001, format="%.5f",
        help="Amplitude Perturbation Quotient over 5 cycles.")
    MDVP_APQ = st.number_input("MDVP:APQ", value=0.02408, step=0.001, format="%.5f",
        help="General measure of amplitude variation.")
    Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.04762, step=0.001, format="%.5f",
        help="Derived from APQ3.")

    st.subheader("Other Voice Quality Features")

    NHR = st.number_input("NHR (Noise-to-Harmonics Ratio)", value=0.01764, step=0.001, format="%.5f",
        help="Ratio of noise to harmonic components. Higher values may indicate disorder.")
    HNR = st.slider("HNR (Harmonics-to-Noise Ratio)", 5.0, 35.0, 21.88, 0.1,
        help="Ratio of harmonic sound to noise. Higher = clearer voice.")
    RPDE = st.number_input("RPDE", value=0.498, step=0.01, format="%.4f",
        help="Recurrence Period Density Entropy – measures complexity in voice signal.")
    DFA = st.number_input("DFA", value=0.682, step=0.01, format="%.4f",
        help="Detrended Fluctuation Analysis – measures self-similarity in voice.")
    spread1 = st.number_input("spread1", value=-5.684, step=0.1, format="%.3f",
        help="Nonlinear measure related to frequency variation.")
    spread2 = st.number_input("spread2", value=0.226, step=0.01, format="%.4f",
        help="Another nonlinear measure of frequency variation.")
    D2 = st.number_input("D2", value=2.381, step=0.1, format="%.3f",
        help="Correlation dimension – measures complexity of the voice signal.")
    PPE = st.number_input("PPE", value=0.207, step=0.01, format="%.4f",
        help="Pitch Period Entropy – measures disorder in pitch periods.")

# Create input DataFrame
input_data = pd.DataFrame({
    'MDVP:Fo(Hz)': [MDVP_Fo],
    'MDVP:Fhi(Hz)': [MDVP_Fhi],
    'MDVP:Flo(Hz)': [MDVP_Flo],
    'MDVP:Jitter(%)': [MDVP_Jitter],
    'MDVP:Jitter(Abs)': [MDVP_Jitter_Abs],
    'MDVP:RAP': [MDVP_RAP],
    'MDVP:PPQ': [MDVP_PPQ],
    'Jitter:DDP': [Jitter_DDP],
    'MDVP:Shimmer': [MDVP_Shimmer],
    'MDVP:Shimmer(dB)': [MDVP_Shimmer_dB],
    'Shimmer:APQ3': [Shimmer_APQ3],
    'Shimmer:APQ5': [Shimmer_APQ5],
    'MDVP:APQ': [MDVP_APQ],
    'Shimmer:DDA': [Shimmer_DDA],
    'NHR': [NHR],
    'HNR': [HNR],
    'RPDE': [RPDE],
    'DFA': [DFA],
    'spread1': [spread1],
    'spread2': [spread2],
    'D2': [D2],
    'PPE': [PPE]
})

st.markdown("---")

# ==================== PREDICTION ====================
if st.button("🔍 Predict", type="primary", use_container_width=True):
    prediction = model.predict(input_data)
    prob_parkinsons = model.predict_proba(input_data)[0][1]
    prob_healthy = 1 - prob_parkinsons

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("**Parkinson's Disease Detected**")
    else:
        st.success("**Healthy (No Parkinson's Detected)**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probability of Parkinson's", f"{prob_parkinsons * 100:.2f}%")
    with col_b:
        st.metric("Probability of Healthy", f"{prob_healthy * 100:.2f}%")

    st.progress(float(prob_parkinsons), text=f"Parkinson's Probability: {prob_parkinsons * 100:.1f}%")

    st.caption("⚠️ This is an AI prediction for educational purposes only. Please consult a doctor.")

    # ==================== SHAP EXPLANATION ====================
    st.markdown("---")
    if st.button("📊 Explain Prediction with SHAP", use_container_width=True):
        with st.spinner("Calculating SHAP values... This may take a few seconds"):
            # Get components from pipeline
            scaler = model.named_steps['scaler']
            xgb_model = model.named_steps['classifier']

            # Scale the input
            input_scaled = scaler.transform(input_data)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(input_scaled)

            # Create Explanation object
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_scaled[0],
                feature_names=input_data.columns.tolist()
            )

            # Plot waterfall
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)

            st.caption(
                "🔴 Red bars = Features that **increase** the chance of Parkinson's\n"
                "🔵 Blue bars = Features that **decrease** the chance of Parkinson's"
            )

# Feature Importance
st.markdown("---")

with st.expander("📊 Show Feature Importance"):
    xgb_model = model.named_steps['classifier']
    importances = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    st.bar_chart(importances.set_index('Feature'))
    st.caption("Higher bars = More important features according to the model")

st.markdown("---")
st.caption("Parkinson's Detection App | Educational Project")