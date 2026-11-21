class TestPredictEndpoint:
    def test_predict_valid_patient(self, client, sample_patient):
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 200
        data = response.get_json()
        assert "logit" in data
        assert "probability" in data
        assert "prediction" in data
        assert "label" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] in (0, 1)
        assert data["label"] in ("Maladie cardiaque", "Pas de maladie")

    def test_predict_missing_features(self, client):
        response = client.post("/predict", json={"age": 55})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_predict_empty_body(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_predict_probability_range(self, client, sample_patient):
        response = client.post("/predict", json=sample_patient)
        data = response.get_json()
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] == int(data["probability"] >= 0.5)
