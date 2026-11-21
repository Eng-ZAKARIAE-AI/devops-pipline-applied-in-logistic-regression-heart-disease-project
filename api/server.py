import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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


def _predict_from_dict(data):
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({"error": f"Features manquantes : {missing}"}), 400

    x = np.array([[float(data[f]) for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)
    z = x_scaled @ w + b
    p = sigmoid(z)
    pred = int(p[0] >= 0.5)

    return jsonify(
        {
            "logit": round(float(z[0]), 4),
            "probability": round(float(p[0]), 4),
            "prediction": pred,
            "label": "Maladie cardiaque" if pred == 1 else "Pas de maladie",
        }
    )


@app.route("/predict", methods=["POST", "GET"])
def predict():
    try:
        if request.method == "GET":
            data = {k: v for k, v in request.args.items()}
        else:
            data = request.get_json(force=True)
        return _predict_from_dict(data)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
