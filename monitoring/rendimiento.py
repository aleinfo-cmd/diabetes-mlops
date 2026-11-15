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
UMBRAL_PRECISION = 0.50
MAX_CAMBIO_POSITIVOS = 0.15

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


def alertas(nombre, r, base):
    avisos = []
    if r["recall"] < UMBRAL_RECALL:
        avisos.append(f"recall {r['recall']:.3f} < {UMBRAL_RECALL}")
    if r["precision"] < UMBRAL_PRECISION:
        avisos.append(f"precision {r['precision']:.3f} < {UMBRAL_PRECISION}")
    cambio = abs(r["positivos_predichos"] - base["positivos_predichos"])
    if cambio > MAX_CAMBIO_POSITIVOS:
        avisos.append(
            f"proporcion de positivos predichos cambio {cambio:.3f} "
            f"(base {base['positivos_predichos']:.3f} -> {r['positivos_predichos']:.3f})"
        )
    return avisos


resultados = {"base": evaluar(X_base), "con_drift": evaluar(X_drift)}

print(f"{'metrica':22} {'base':>8} {'con drift':>10}")
for k in resultados["base"]:
    print(f"{k:22} {resultados['base'][k]:>8.3f} {resultados['con_drift'][k]:>10.3f}")

print()
for nombre, r in resultados.items():
    avisos = alertas(nombre, r, resultados["base"])
    if avisos:
        for a in avisos:
            print(f"ALERTA ({nombre}): {a}")
    else:
        print(f"OK ({nombre}): sin alertas")

with open("monitoring/reporte_rendimiento.json", "w") as f:
    json.dump(resultados, f, indent=2)