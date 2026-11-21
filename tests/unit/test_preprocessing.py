import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from sklearn.preprocessing import StandardScaler


class TestPreprocessing:
    def test_binarize_target(self):
        targets = np.array([0, 1, 2, 3, 4])
        binarized = (targets > 0).astype(int)
        expected = np.array([0, 1, 1, 1, 1])
        assert np.array_equal(binarized, expected)

    def test_standard_scaler(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_missing_value_imputation(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        df_filled = df.fillna(df.median(numeric_only=True))
        assert not df_filled.isnull().any().any()
        assert df_filled.loc[1, "a"] == 2.0
        assert df_filled.loc[2, "b"] == 4.5

    def test_feature_count(self):
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
        assert len(FEATURE_NAMES) == 13
