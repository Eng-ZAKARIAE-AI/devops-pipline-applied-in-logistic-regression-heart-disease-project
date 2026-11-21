import os
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.train import sigmoid, train_logistic_regression


class TestFullPipeline:
    def test_training_produces_weights(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        y = np.array([0, 0, 0, 1])
        w, b, loss = train_logistic_regression(X, y, lr=0.5, epochs=5000, verbose=False)
        assert w.shape == (2,)
        assert isinstance(b, float)
        assert loss > 0

    def test_training_improves_accuracy(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        y = np.array([0, 0, 0, 1])
        w, b, _ = train_logistic_regression(X, y, lr=0.5, epochs=5000, verbose=False)
        preds = (sigmoid(X @ w + b) >= 0.5).astype(int)
        acc = (preds == y).mean()
        assert acc >= 0.75

    def test_model_artifacts_loadable(self):
        for fname in ["model_weights.pkl", "model_bias.pkl", "scaler.pkl"]:
            path = os.path.join(ROOT, "artifacts", fname)
            assert os.path.exists(path), f"{fname} missing"
            obj = joblib.load(path)
            assert obj is not None

    def test_prediction_end_to_end(self, client, sample_patient):
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data
        assert "probability" in data
