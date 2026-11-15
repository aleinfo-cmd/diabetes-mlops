import json
import sys

import pandas as pd
from scipy.stats import ks_2samp

ALFA = 0.05

referencia = pd.read_csv("data/processed/train.csv").drop(columns="Outcome")
actual = pd.read_csv(sys.argv[1])

resultados = {}
for col in referencia.columns:
    ks, p = ks_2samp(referencia[col].dropna(), actual[col].dropna())
    resultados[col] = {
        "ks": round(float(ks), 3),
        "p_value": round(float(p), 4),
        "drift": bool(p < ALFA),
    }

print(f"Archivo analizado: {sys.argv[1]}\n")
print(f"{'variable':28} {'KS':>6} {'p-value':>9}  drift")
for col, r in resultados.items():
    marca = "SI" if r["drift"] else "no"
    print(f"{col:28} {r['ks']:>6} {r['p_value']:>9}  {marca}")

con_drift = [c for c, r in resultados.items() if r["drift"]]
print(f"\nVariables con drift: {con_drift if con_drift else 'ninguna'}")

with open("monitoring/reporte_drift.json", "w") as f:
    json.dump(resultados, f, indent=2)