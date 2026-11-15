# diabetes-mlops

Pipeline MLOps para predecir diabetes (dataset Pima Indians) con Random Forest.

## Estructura del proyecto

| Carpeta / archivo | Contenido |
|---|---|
| `data/` | Datos versionados con DVC (`raw` y `processed`) |
| `src/` | Preprocesamiento y entrenamiento |
| `models/` | Modelo exportado para la API |
| `api/` | API REST con FastAPI |
| `dags/` | DAG de Airflow |
| `airflow/` | Imagen Docker de Airflow |
| `monitoring/` | Detección de drift y rendimiento |
| `tests/` | Pruebas automáticas |
| `.github/workflows/` | CI/CD con GitHub Actions |
| `Dockerfile` | Imagen de la API |
| `dvc.yaml`, `params.yaml` | Pipeline y parámetros |