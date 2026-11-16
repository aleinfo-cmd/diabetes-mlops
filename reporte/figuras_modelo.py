from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FIG = Path("reporte/figuras")
TAB = Path("reporte/tablas")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

model = joblib.load("models/model.joblib")
raw = pd.read_csv("data/raw/diabetes.csv")
test = pd.read_csv("data/processed/test.csv")

X_test = test.drop(columns="Outcome").astype("float64")
y_test = test["Outcome"]
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / nombre, dpi=200)
    plt.close(fig)
    print("Guardado:", FIG / nombre)


# 1. Distribución de la variable objetivo
conteo = raw["Outcome"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
barras = ax.bar(
    ["Sin diabetes (0)", "Diabetes (1)"], conteo.values, color=["#4C72B0", "#DD8452"]
)
ax.bar_label(barras)
ax.set_ylabel("Pacientes")
ax.set_title("Distribución de la variable objetivo")
guardar(fig, "01_distribucion_clases.png")

# 2. Ceros imposibles (datos faltantes disfrazados)
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ceros = (raw[cols] == 0).sum()
fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar(cols, ceros.values, color="#C44E52")
ax.bar_label(barras)
ax.set_ylabel("Registros con valor 0")
ax.set_title("Ceros imposibles tratados como datos faltantes")
ax.tick_params(axis="x", rotation=30)
guardar(fig, "02_ceros_imposibles.png")

# 3. Matriz de confusión
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    display_labels=["Sin diabetes", "Diabetes"],
    cmap="Blues",
    ax=ax,
)
ax.set_title("Matriz de confusión (conjunto de prueba)")
guardar(fig, "03_matriz_confusion.png")

# 4. Curva ROC
fig, ax = plt.subplots(figsize=(5, 4.5))
RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="Random Forest")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
ax.legend()
ax.set_title("Curva ROC")
guardar(fig, "04_curva_roc.png")

# 5. Importancia de variables
imp = pd.Series(
    model.named_steps["clf"].feature_importances_, index=X_test.columns
).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp.index, imp.values, color="#55A868")
ax.set_xlabel("Importancia")
ax.set_title("Importancia de variables (Random Forest)")
guardar(fig, "05_importancia_variables.png")

# 6. Métricas del modelo
metricas = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1-score": f1_score(y_test, pred),
    "ROC-AUC": roc_auc_score(y_test, proba),
}
tabla = pd.DataFrame(
    {"metrica": list(metricas), "valor": [round(v, 3) for v in metricas.values()]}
)
tabla.to_csv(TAB / "tabla_metricas.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar(tabla["metrica"], tabla["valor"], color="#8172B3")
ax.bar_label(barras)
ax.set_ylim(0, 1)
ax.set_title("Métricas en el conjunto de prueba")
guardar(fig, "06_metricas_modelo.png")

# Tabla descriptiva del dataset
desc = raw.describe().T.round(2).rename_axis("variable")
desc.to_csv(TAB / "tabla_descriptiva.csv")

print("\nMétricas del modelo:")
print(tabla.to_string(index=False))
print("\nEstadística descriptiva:")
print(desc.to_string())