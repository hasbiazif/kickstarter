import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load preprocessor & model
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/best_model.pkl")

st.title("🚀 Kickstarter Success Prediction App")
st.write("Masukkan detail proyek untuk memprediksi peluang sukses di Kickstarter.")

# --- Input form ---
usd_goal = st.number_input("Target Dana (USD)", min_value=100, max_value=1_000_000, step=100, value=5000)
duration = st.slider("Durasi Kampanye (hari)", min_value=1, max_value=90, value=30)
title_len = st.slider("Panjang Judul (karakter)", min_value=5, max_value=85, value=35)
blurb_len = st.slider("Panjang Blurb (karakter)", min_value=20, max_value=150, value=100)

# Dropdown kategori (parent category)
categories = [
    "Art", "Comics", "Crafts", "Dance", "Design", "Fashion", "Film & Video", 
    "Food", "Games", "Journalism", "Music", "Photography", "Publishing", 
    "Technology", "Theater"
]
category = st.selectbox("Kategori Utama", categories, index=categories.index("Technology"))

# Dropdown country (beberapa kode populer di Kickstarter)
countries = ["US", "GB", "CA", "AU", "DE", "FR", "NL", "SE", "IT", "ES", "Other"]
country = st.selectbox("Negara", countries, index=0)

# --- Siapkan data untuk prediksi ---
usd_goal_log = np.log1p(usd_goal)  # log transform

input_data = pd.DataFrame([{
    "usd_goal_log": usd_goal_log,
    "duration_days": duration,
    "title_len": title_len,
    "blurb_len": blurb_len,
    "category_parent": category,
    "country": country
}])

# --- Prediksi ---
if st.button("Prediksi"):
    X_new = preprocessor.transform(input_data)
    prob = model.predict_proba(X_new)[:, 1][0]
    label = "Sukses ✅" if prob >= 0.5 else "Gagal ❌"

    st.subheader(f"Hasil Prediksi: {label}")
    st.write(f"Probabilitas sukses: **{prob:.2f}**")