import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. CHARGEMENT DES DONNÉES RÉELLES (UCI Heart Disease - Cleveland)
# ============================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
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
    "target",
]

df = pd.read_csv(url, names=columns, na_values="?")
print("=== DONNÉES BRUTES ===")
print(f"Forme : {df.shape}")
print(f"Valeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================================
# 2. PRÉTRAITEMENT
# ============================================================
# Gestion des valeurs manquantes (médiane)
df.fillna(df.median(numeric_only=True), inplace=True)

# Binariser le target (0 = pas de maladie, 1-4 = maladie)
df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1).values
y = df["target"].values

# Normalisation (StandardScaler) — ESSENTIEL pour la régression logistique
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test (80/20, stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# 3. RÉGRESSION LOGISTIQUE — DEPUIS ZÉRO (From Scratch)
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.1, epochs=2000, verbose=True):
    """
    Pipeline : x → z = x·w + b (LOGIT) → σ(z) (PROBABILITÉ) → classe
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        # --- FORWARD ---
        z = X @ w + b  # ← LOGITS (scores bruts)
        y_pred = sigmoid(z)  # ← PROBABILITÉS

        # --- LOSS (Binary Cross-Entropy) ---
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(
            y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped)
        )

        # --- BACKWARD ---
        error = y_pred - y
        dw = (1 / n_samples) * (X.T @ error)
        db = (1 / n_samples) * np.sum(error)

        # --- UPDATE ---
        w -= lr * dw
        b -= lr * db

        if verbose and epoch % 500 == 0:
            acc = ((sigmoid(X @ w + b) >= 0.5).astype(int) == y).mean()
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Acc train: {acc:.3f}")

    return w, b, loss


print("\n=== ENTRAÎNEMENT (from scratch) ===")
w, b, final_loss = train_logistic_regression(X_train, y_train, lr=0.1, epochs=2000)

# ============================================================
# 4. ÉVALUATION & COMPARAISON SKLEARN
# ============================================================
z_test = X_test @ w + b
p_test = sigmoid(z_test)
y_pred_test = (p_test >= 0.5).astype(int)

print(f"\nAccuracy (From scratch) : {accuracy_score(y_test, y_pred_test):.4f}")
print(
    classification_report(
        y_test, y_pred_test, target_names=["Pas de maladie", "Maladie"]
    )
)

sk_model = LogisticRegression(max_iter=5000, C=10)
sk_model.fit(X_train, y_train)
print(
    f"Accuracy (Sklearn)      : {accuracy_score(y_test, sk_model.predict(X_test)):.4f}"
)

# ============================================================
# 5. SAUVEGARDE DU MODÈLE + SCALER
# ============================================================
joblib.dump(w, "artifacts/model_weights.pkl")
joblib.dump(b, "artifacts/model_bias.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
print("\n✅ Modèle sauvegardé : model_weights.pkl, model_bias.pkl, scaler.pkl")
