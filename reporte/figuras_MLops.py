import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import ks_2samp

FIG = Path("reporte/figuras")
TAB = Path("reporte/tablas")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ALFA = 0.05
UMBRAL_RECALL = 0.70
UMBRAL_PRECISION = 0.50
MAX_CAMBIO_POSITIVOS = 0.15

ref = pd.read_csv("data/processed/train.csv").drop(columns="Outcome")
normal = pd.read_csv("monitoring/produccion_normal.csv")
drift = pd.read_csv("monitoring/produccion_con_drift.csv")
rend = json.load(open("monitoring/reporte_rendimiento.json"))


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / nombre, dpi=200)
    plt.close(fig)
    print("Guardado:", FIG / nombre)


# ---------- Tabla y gráfica de drift ----------
filas = []
for col in ref.columns:
    ks_n, p_n = ks_2samp(ref[col].dropna(), normal[col].dropna())
    ks_d, p_d = ks_2samp(ref[col].dropna(), drift[col].dropna())
    filas.append(
        {
            "variable": col,
            "ks_normal": round(ks_n, 3),
            "p_normal": round(p_n, 4),
            "ks_drift": round(ks_d, 3),
            "p_drift": round(p_d, 4),
            "drift_detectado": bool(p_d < ALFA),
        }
    )
tabla_drift = pd.DataFrame(filas)
tabla_drift.to_csv(TAB / "tabla_drift.csv", index=False)

x = np.arange(len(tabla_drift))
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(x - 0.2, tabla_drift["ks_normal"], 0.4, label="Datos normales", color="#4C72B0")
barras = ax.bar(
    x + 0.2, tabla_drift["ks_drift"], 0.4, label="Datos con drift", color="#DD8452"
)
for b, det in zip(barras, tabla_drift["drift_detectado"]):
    if det:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            "*",
            ha="center",
            fontsize=16,
            color="#C44E52",
        )
ax.set_xticks(x)
ax.set_xticklabels(tabla_drift["variable"], rotation=30, ha="right")
ax.set_ylabel("Estadístico KS")
ax.set_title("Data drift por variable (* = drift detectado, p < 0.05)")
ax.legend()
guardar(fig, "07_drift_por_variable.png")

# ---------- Distribuciones: entrenamiento vs datos con drift ----------
variables = ["Glucose", "BMI", "Age"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, variables):
    ax.hist(ref[col].dropna(), bins=25, alpha=0.6, density=True, label="Entrenamiento")
    ax.hist(drift[col].dropna(), bins=25, alpha=0.6, density=True, label="Con drift")
    ax.set_title(col)
    ax.set_ylabel("Densidad")
axes[0].legend()
fig.suptitle("Distribución de las variables con drift")
guardar(fig, "08_distribuciones_drift.png")

# ---------- Rendimiento: base vs con drift ----------
nombres = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "positivos_predichos": "Prop. positivos\npredichos",
}
claves = list(nombres)
base = [rend["base"][k] for k in claves]
con = [rend["con_drift"][k] for k in claves]
x = np.arange(len(claves))
fig, ax = plt.subplots(figsize=(10, 4.8))
b1 = ax.bar(x - 0.2, base, 0.4, label="Datos base", color="#4C72B0")
b2 = ax.bar(x + 0.2, con, 0.4, label="Datos con drift", color="#DD8452")
ax.bar_label(b1, fmt="%.2f", fontsize=8)
ax.bar_label(b2, fmt="%.2f", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([nombres[k] for k in claves])
ax.set_ylim(0, 1.15)
ax.set_title("Rendimiento del modelo: datos base vs datos con drift")
ax.legend()
guardar(fig, "09_rendimiento_base_vs_drift.png")

# ---------- Tabla de alertas ----------
b, d = rend["base"], rend["con_drift"]
cambio = abs(d["positivos_predichos"] - b["positivos_predichos"])
alertas = pd.DataFrame(
    [
        {
            "regla": f"Recall >= {UMBRAL_RECALL}",
            "base": round(b["recall"], 3),
            "con_drift": round(d["recall"], 3),
            "estado_base": "OK" if b["recall"] >= UMBRAL_RECALL else "ALERTA",
            "estado_drift": "OK" if d["recall"] >= UMBRAL_RECALL else "ALERTA",
        },
        {
            "regla": f"Precision >= {UMBRAL_PRECISION}",
            "base": round(b["precision"], 3),
            "con_drift": round(d["precision"], 3),
            "estado_base": "OK" if b["precision"] >= UMBRAL_PRECISION else "ALERTA",
            "estado_drift": "OK" if d["precision"] >= UMBRAL_PRECISION else "ALERTA",
        },
        {
            "regla": f"Cambio en positivos predichos <= {MAX_CAMBIO_POSITIVOS}",
            "base": 0.0,
            "con_drift": round(cambio, 3),
            "estado_base": "OK",
            "estado_drift": "OK" if cambio <= MAX_CAMBIO_POSITIVOS else "ALERTA",
        },
    ]
)
alertas.to_csv(TAB / "tabla_alertas.csv", index=False)


# ---------- Diagrama del pipeline ----------
def caja(ax, x, y, w, h, texto, color, borde="#333333", estilo="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=color,
            edgecolor=borde,
            linestyle=estilo,
            linewidth=1.4,
        )
    )
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=9)


def flecha(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4)
    )


fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.5)
ax.axis("off")

caja(ax, 0.2, 2.7, 1.9, 1.0, "Datos\n(DVC)", "#D6E4F0")
caja(ax, 2.5, 2.7, 1.9, 1.0, "Preprocesamiento\n(src/preprocess.py)", "#D6E4F0")
caja(ax, 4.8, 2.7, 1.9, 1.0, "Entrenamiento\nRandom Forest", "#D6E4F0")
caja(ax, 7.1, 4.9, 1.9, 1.0, "MLflow\nexperimentos y registro", "#E5DDF0")
flecha(ax, 6.4, 3.7, 7.4, 4.9)
ax.text(8.05, 3.35, "models/model.joblib", ha="center", fontsize=8, style="italic")
caja(ax, 9.4, 2.7, 2.2, 1.0, "API REST\n(FastAPI)", "#DDEFD9")
for xa, xb in [(2.1, 2.5), (4.4, 4.8), (6.7, 9.4)]:
    flecha(ax, xa, 3.2, xb, 3.2)

caja(
    ax, 2.5, 4.9, 4.2, 1.0,
    "Apache Airflow (DAG semanal)\npreprocess → train → verificar_metricas",
    "#FFF3CD", estilo="--",
)
flecha(ax, 3.5, 4.9, 3.5, 3.8)
flecha(ax, 5.7, 4.9, 5.7, 3.8)

caja(ax, 0.2, 0.8, 1.9, 1.0, "Git / GitHub\ncódigo y ramas", "#EEEEEE")
flecha(ax, 2.1, 1.3, 2.5, 1.3)
caja(ax, 9.4, 4.9, 2.2, 1.0, "Monitoreo\ndrift (KS) y rendimiento", "#F8D7DA")
flecha(ax, 10.5, 3.7, 10.5, 4.9)

caja(ax, 9.4, 0.8, 2.2, 1.0, "Docker\nimagen en GHCR", "#DDEFD9")
flecha(ax, 10.5, 1.8, 10.5, 2.7)
caja(
    ax, 2.5, 0.8, 6.5, 1.0,
    "GitHub Actions (CI/CD)\npruebas → build de Docker → publicar imagen",
    "#EEEEEE", estilo="--",
)
flecha(ax, 9.0, 1.3, 9.4, 1.3)

ax.set_title("Arquitectura del pipeline MLOps", fontsize=13)
guardar(fig, "10_diagrama_pipeline.png")

print("\nTabla de drift:")
print(tabla_drift.to_string(index=False))
print("\nTabla de alertas:")
print(alertas.to_string(index=False))