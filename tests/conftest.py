import os

import joblib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    files = {
        "model_weights.pkl": np.zeros(13),
        "model_bias.pkl": 0.0,
    }
    for fname, obj in files.items():
        path = os.path.join(ARTIFACTS_DIR, fname)
        if not os.path.exists(path):
            joblib.dump(obj, path)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 13)))
        joblib.dump(scaler, scaler_path)


@pytest.fixture
def client():
    from api.server import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_patient():
    return {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 130,
        "chol": 205,
        "fbs": 1,
        "restecg": 0,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 2.3,
        "slope": 1,
        "ca": 2,
        "thal": 3,
    }
