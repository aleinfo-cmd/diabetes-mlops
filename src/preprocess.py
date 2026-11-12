from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

df = pd.read_csv("data/raw/diabetes.csv")

for col in params["zero_as_missing"]:
    df[col] = df[col].replace(0, float("nan"))

train, test = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["Outcome"],
)

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "train.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print(f"train: {train.shape}, test: {test.shape}")