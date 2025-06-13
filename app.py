import streamlit as st
st.set_page_config(page_title="🌾 Crop Production Predictor", layout='centered')

import pandas as pd
import numpy as np
import pickle

# Load model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Load reference data (from your training CSV)
@st.cache_data
def load_data():
    return pd.read_csv("Crop_Production_Cleaned.csv")

df = load_data()
areas = sorted(df["Area"].dropna().unique())
items = sorted(df["Item"].dropna().unique())

# --- Streamlit App UI

st.title("🌾 Crop Production Predictor")
st.markdown("Select options below to predict the crop production value using our ML model.")

# --- Dropdowns instead of text inputs
area = st.selectbox("🌍 Select Area", areas)
item = st.selectbox("🌾 Select Crop Item", items)
year = st.number_input("📅 Enter Year", min_value=1960, max_value=2030, value=2022, step=1)
yield_val = st.number_input("📈 Enter Yield", min_value=0.0)
area_harvested = st.number_input("📐 Enter Area Harvested", min_value=0.0)

# --- Prediction
if st.button("🔍 Predict Production"):
    input_df = pd.DataFrame({
        "Area": [area],
        "Item": [item],
        "Year": [year],
        "Yield": [yield_val],
        "Area harvested": [area_harvested]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Predicted Crop Production: {prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
