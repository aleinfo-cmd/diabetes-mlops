from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

PACIENTE = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": None,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valido():
    r = client.post("/predict", json=PACIENTE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediccion"] in (0, 1)
    assert 0.0 <= body["probabilidad_diabetes"] <= 1.0


def test_predict_con_datos_faltantes():
    datos = {**PACIENTE, "Glucose": None, "BMI": None}
    r = client.post("/predict", json=datos)
    assert r.status_code == 200


def test_predict_edad_invalida():
    r = client.post("/predict", json={**PACIENTE, "Age": 10})
    assert r.status_code == 422