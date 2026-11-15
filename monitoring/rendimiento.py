import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

UMBRAL_RECALL = 0.70

model = joblib.load("models/model.joblib")

test = pd.read_csv("data/processed/test.csv")
y = test["Outcome"]
X_base = test.drop(columns="Outcome").astype("float64")
X_drift = pd.read_csv("monitoring/produccion_con_drift.csv").astype("float64")


def evaluar(X):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "positivos_predichos": float(pred.mean()),
    }


resultados = {"base": evaluar(X_base), "con_drift": evaluar(X_drift)}

print(f"{'metrica':22} {'base':>8} {'con drift':>10}")
for k in resultados["base"]:
    print(f"{k:22} {resultados['base'][k]:>8.3f} {resultados['con_drift'][k]:>10.3f}")

for nombre, r in resultados.items():
    if r["recall"] < UMBRAL_RECALL:
        print(f"\nALERTA ({nombre}): recall {r['recall']:.3f} < {UMBRAL_RECALL}")

with open("monitoring/reporte_rendimiento.json", "w") as f:
    json.dump(resultados, f, indent=2)