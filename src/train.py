import json

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

params = yaml.safe_load(open("params.yaml"))["train"]

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train = train.drop(columns="Outcome").astype("float64")
y_train = train["Outcome"]
X_test = test.drop(columns="Outcome").astype("float64")
y_test = test["Outcome"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes-prediccion")

with mlflow.start_run():
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**params, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_train.head(5),
        registered_model_name="diabetes-clasificador",
        skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
    )

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")