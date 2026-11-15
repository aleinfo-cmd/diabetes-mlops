FROM apache/airflow:2.10.5-python3.11

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

RUN python -m venv /home/airflow/ml-venv && \
    /home/airflow/ml-venv/bin/pip install --no-cache-dir --upgrade pip

RUN /home/airflow/ml-venv/bin/pip install --no-cache-dir \
    numpy==2.4.6 pandas==3.1.1 scikit-learn==1.9.1 joblib==1.6.0 pyyaml==6.0.3

RUN /home/airflow/ml-venv/bin/pip install --no-cache-dir mlflow==3.18.0