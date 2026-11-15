import pandas as pd

test = pd.read_csv("data/processed/test.csv").drop(columns="Outcome")

normal = test.sample(n=len(test), random_state=1)
normal.to_csv("monitoring/produccion_normal.csv", index=False)

con_drift = test.copy()
con_drift["Glucose"] = con_drift["Glucose"] + 35
con_drift["BMI"] = con_drift["BMI"] + 6
con_drift["Age"] = con_drift["Age"] + 12
con_drift.to_csv("monitoring/produccion_con_drift.csv", index=False)

print("Archivos generados en monitoring/")