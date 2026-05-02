import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model and labels
model = pickle.load(open('crop_model.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

st.title("🌾 Crop Recommendation System")

st.write("Enter soil and weather values:")

# Inputs
N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
P = st.number_input("Phosphorus (P)", 0.0, 150.0, 40.0)
K = st.number_input("Potassium (K)", 0.0, 150.0, 40.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

# Predict button
if st.button("Predict Crop"):

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N','P','K','temperature','humidity','ph','rainfall'])

    probs = model.predict_proba(input_data)[0]

    # Top 3 crops
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_crops = [labels[i] for i in top3_idx]
    top3_probs = probs[top3_idx]

    st.subheader("🌾 Top 3 Recommended Crops")

    for i in range(3):
        st.write(f"{i+1}. {top3_crops[i]} ({top3_probs[i]:.2f})")

    # ---------------------------
    # 📊 Bar Chart
    # ---------------------------
    st.subheader("📊 Prediction Confidence")

    fig, ax = plt.subplots()
    ax.bar(top3_crops, top3_probs)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Crops")

    st.pyplot(fig)

    # ---------------------------
    # 🌱 Soil Health Analysis
    # ---------------------------
    st.subheader("🌱 Soil Health Analysis")

    issues = []
    actions = []

    if N < 50:
        issues.append("Nitrogen is low")
        actions.append("Apply urea")
    elif N > 120:
        issues.append("Nitrogen is high")
        actions.append("Reduce nitrogen fertilizer")

    if P < 40:
        issues.append("Phosphorus is low")
        actions.append("Add DAP")
    elif P > 100:
        issues.append("Phosphorus is high")
        actions.append("Reduce phosphorus")

    if K < 40:
        issues.append("Potassium is low")
        actions.append("Apply potash")
    elif K > 80:
        issues.append("Potassium is high")
        actions.append("Reduce potassium")

    if ph < 5.5:
        issues.append("Soil is acidic")
        actions.append("Apply lime")
    elif ph > 7:
        issues.append("Soil is alkaline")
        actions.append("Add organic matter")

    if issues:
        st.warning("⚠️ Issues Found:")
        for i in issues:
            st.write("-", i)
    else:
        st.success("✅ Soil condition is good")

    st.subheader("🌿 Recommendations")
    for a in actions:
        st.write("-", a)
