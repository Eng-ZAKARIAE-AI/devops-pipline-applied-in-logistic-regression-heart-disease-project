import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# --- Chargement du modèle AU DÉMARRAGE ---
w = joblib.load("artifacts/model_weights.pkl")
b = joblib.load("artifacts/model_bias.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validation des champs
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Features manquantes : {missing}"}), 400

        # Vecteur de features dans l'ordre exact de l'entraînement
        x = np.array([[data[f] for f in FEATURE_NAMES]])

        # --- PIPELINE COMPLET ---
        x_scaled = scaler.transform(x)  # 1. Normalisation
        z = x_scaled @ w + b  # 2. LOGIT
        p = sigmoid(z)  # 3. PROBABILITÉ
        pred = int(p[0] >= 0.5)  # 4. CLASSE

        return jsonify(
            {
                "logit": round(float(z[0]), 4),
                "probability": round(float(p[0]), 4),
                "prediction": pred,
                "label": "Maladie cardiaque" if pred == 1 else "Pas de maladie",
            }
        )

    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
