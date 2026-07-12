import streamlit as st
import joblib
import pandas as pd

model = joblib.load('parkinsons_xgboost_tuned_pipeline.pkl')

st.set_page_config(page_title="Parkinson's Detection", page_icon="🧠", layout="centered")

st.title("🧠 Parkinson's Disease Detection")
st.markdown("### Voice-based Parkinson's Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Frequency Features")
    MDVP_Fo = st.slider("MDVP:Fo(Hz)", 80.0, 300.0, 154.23, 0.1)
    MDVP_Fhi = st.slider("MDVP:Fhi(Hz)", 100.0, 600.0, 197.10, 0.1)
    MDVP_Flo = st.slider("MDVP:Flo(Hz)", 60.0, 250.0, 116.32, 0.1)

    st.subheader("Jitter Features")
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", value=0.00622, step=0.0001, format="%.5f")
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", value=0.00004, step=0.00001, format="%.6f")
    MDVP_RAP = st.number_input("MDVP:RAP", value=0.00331, step=0.0001, format="%.5f")
    MDVP_PPQ = st.number_input("MDVP:PPQ", value=0.00345, step=0.0001, format="%.5f")
    Jitter_DDP = st.number_input("Jitter:DDP", value=0.00994, step=0.0001, format="%.5f")

with col2:
    st.subheader("Shimmer Features")
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.02971, step=0.001, format="%.5f")
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.282, step=0.01, format="%.3f")
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.01587, step=0.001, format="%.5f")
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.01809, step=0.001, format="%.5f")
    MDVP_APQ = st.number_input("MDVP:APQ", value=0.02408, step=0.001, format="%.5f")
    Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.04762, step=0.001, format="%.5f")

    st.subheader("Other Features")
    NHR = st.number_input("NHR", value=0.01764, step=0.001, format="%.5f")
    HNR = st.slider("HNR", 5.0, 35.0, 21.88, 0.1)
    RPDE = st.number_input("RPDE", value=0.498, step=0.01, format="%.4f")
    DFA = st.number_input("DFA", value=0.682, step=0.01, format="%.4f")
    spread1 = st.number_input("spread1", value=-5.684, step=0.1, format="%.3f")
    spread2 = st.number_input("spread2", value=0.226, step=0.01, format="%.4f")
    D2 = st.number_input("D2", value=2.381, step=0.1, format="%.3f")
    PPE = st.number_input("PPE", value=0.207, step=0.01, format="%.4f")

input_data = pd.DataFrame({
    'MDVP:Fo(Hz)': [MDVP_Fo], 'MDVP:Fhi(Hz)': [MDVP_Fhi], 'MDVP:Flo(Hz)': [MDVP_Flo],
    'MDVP:Jitter(%)': [MDVP_Jitter], 'MDVP:Jitter(Abs)': [MDVP_Jitter_Abs],
    'MDVP:RAP': [MDVP_RAP], 'MDVP:PPQ': [MDVP_PPQ], 'Jitter:DDP': [Jitter_DDP],
    'MDVP:Shimmer': [MDVP_Shimmer], 'MDVP:Shimmer(dB)': [MDVP_Shimmer_dB],
    'Shimmer:APQ3': [Shimmer_APQ3], 'Shimmer:APQ5': [Shimmer_APQ5],
    'MDVP:APQ': [MDVP_APQ], 'Shimmer:DDA': [Shimmer_DDA],
    'NHR': [NHR], 'HNR': [HNR], 'RPDE': [RPDE], 'DFA': [DFA],
    'spread1': [spread1], 'spread2': [spread2], 'D2': [D2], 'PPE': [PPE]
})

st.markdown("---")

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

with st.expander("📊 Show Feature Importance"):
    xgb_model = model.named_steps['classifier']
    importances = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    st.bar_chart(importances.set_index('Feature'))
    st.caption("Higher = More important for prediction")

st.markdown("---")
st.caption("Parkinson's Detection App | Built with XGBoost")