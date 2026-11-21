import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000/predict")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:5000/health")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

st.title("🫀 Heart Disease Risk Prediction")
st.caption(
    "Logistic Regression (From Scratch) • Pipeline : Features → Logit (z) → Sigmoid σ(z) → Probability"
)

# Vérification de l'état de l'API Flask
try:
    health = requests.get(HEALTH_URL, timeout=2)
    if health.status_code == 200:
        st.success("API Backend connectée (port 5000)")
    else:
        st.warning("API détectée mais statut anormal.")
except requests.exceptions.RequestException:
    st.error(
        "⚠️ API Flask hors ligne ! Lance d'abord `make run` dans un autre terminal."
    )

st.subheader("📋 Paramètres cliniques du patient")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Âge (ans)", 20, 90, 52)
    sex = st.selectbox(
        "Sexe",
        options=[1, 0],
        format_func=lambda x: "Homme (1)" if x == 1 else "Femme (0)",
    )
    cp = st.selectbox(
        "Type de douleur thoracique (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Asymptomatique",
            1: "1 - Angine typique",
            2: "2 - Angine atypique",
            3: "3 - Douleur non-angineuse",
        }[x],
    )
    trestbps = st.number_input("Tension au repos (trestbps, mm Hg)", 80, 220, 125)
    chol = st.number_input("Cholestérol (chol, mg/dl)", 100, 600, 212)
    fbs = st.selectbox(
        "Glycémie à jeun > 120 mg/dl (fbs)",
        options=[0, 1],
        format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)",
    )
    restecg = st.selectbox("ECG au repos (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.slider("Fréquence cardiaque max (thalach)", 60, 220, 168)
    exang = st.selectbox(
        "Angine induite par l'effort (exang)",
        options=[0, 1],
        format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)",
    )
    oldpeak = st.number_input(
        "Dépression ST à l'effort (oldpeak)", 0.0, 7.0, 1.0, step=0.1
    )
    slope = st.selectbox("Pente du segment ST (slope)", options=[0, 1, 2], index=2)
    ca = st.selectbox("Vaisseaux majeurs colorés (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassémie (thal)", options=[1, 2, 3], index=1)

if st.button("🔍 Lancer la prédiction", type="primary", use_container_width=True):
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        result = response.json()

        if response.status_code == 200:
            st.divider()
            st.subheader("📊 Résultat du modèle")

            m1, m2, m3 = st.columns(3)
            m1.metric("Logit brut (z)", f"{result['logit']:.4f}")
            m2.metric("Probabilité σ(z)", f"{result['probability'] * 100:.2f} %")
            m3.metric("Classe prédite", result["prediction"])

            st.progress(float(result["probability"]))

            if result["prediction"] == 1:
                st.error(f"🚨 Diagnostic : **{result['label']}**")
            else:
                st.success(f"✅ Diagnostic : **{result['label']}**")
        else:
            st.error(f"Erreur API : {result.get('error', 'Erreur inconnue')}")

    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de joindre l'API Flask ({API_URL}) : {e}")
