import requests

url = "http://localhost:5000/predict"

patient = {
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

response = requests.post(url, json=patient)
result = response.json()

print(f"Logit        : {result['logit']}")
print(f"Probabilité  : {result['probability']}")
print(f"Prédiction   : {result['prediction']}")
print(f"Diagnostic   : {result['label']}")
