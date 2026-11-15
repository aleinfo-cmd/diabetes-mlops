import json
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROYECTO = "/opt/airflow/proyecto"
PYTHON = "/home/airflow/ml-venv/bin/python"
UMBRAL_RECALL = 0.70


def verificar_metricas():
    with open(f"{PROYECTO}/metrics.json") as f:
        metricas = json.load(f)
    print("Métricas del modelo:", metricas)
    if metricas["recall"] < UMBRAL_RECALL:
        raise ValueError(
            f"Recall {metricas['recall']:.3f} por debajo del umbral {UMBRAL_RECALL}"
        )


with DAG(
    dag_id="diabetes_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "diabetes"],
) as dag:
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"cd {PROYECTO} && {PYTHON} src/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        bash_command=f"cd {PROYECTO} && {PYTHON} src/train.py",
    )

    verificar = PythonOperator(
        task_id="verificar_metricas",
        python_callable=verificar_metricas,
    )

    preprocess >> train >> verificar