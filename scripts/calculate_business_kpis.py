"""Calculate the business KPIs used in the checklist review."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "output" / "modelos" / "support_vector_machine_pipeline.joblib"


def prepare_test_data() -> tuple[pd.DataFrame, pd.Series, int, int]:
    data = pd.read_csv(RAW_DATA)
    total_customers = len(data)
    total_dataset_churn = int((data["Churn"] == "Yes").sum())

    # Match the EDA notebook: remove the identifier and exact duplicate rows
    # before creating the stratified train/test split.
    data = data.drop(columns=["customerID"]).drop_duplicates().reset_index(drop=True)
    _, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["Churn"],
    )

    target = test.pop("Churn").map({"No": 0, "Yes": 1}).astype(int)
    test["TotalCharges"] = pd.to_numeric(test["TotalCharges"], errors="coerce")
    zero_tenure = test["TotalCharges"].isna() & test["tenure"].eq(0)
    test.loc[zero_tenure, "TotalCharges"] = 0.0
    test["TotalCharges"] = test["TotalCharges"].fillna(test["TotalCharges"].median())
    return test, target, total_customers, total_dataset_churn


def main() -> None:
    test, target, total_customers, total_dataset_churn = prepare_test_data()
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(test)[:, 1]
    predictions = model.predict(test)

    total_churn = int(target.sum())
    true_positives = int(((predictions == 1) & (target == 1)).sum())
    top_count = math.ceil(len(test) * 0.20)
    ranked = pd.DataFrame({"target": target.to_numpy(), "probability": probabilities})
    top_risk = ranked.sort_values("probability", ascending=False).head(top_count)
    captured_churn = int(top_risk["target"].sum())

    print(f"Tasa de churn dataset completo: {total_dataset_churn / total_customers:.4%}")
    print(f"Recall de churn en test: {true_positives / total_churn:.4%}")
    print(
        "Captura de churn en el 20% superior: "
        f"{captured_churn}/{total_churn} = {captured_churn / total_churn:.4%}"
    )
    print(f"Clientes test: {len(test)}; clientes priorizados: {top_count}")


if __name__ == "__main__":
    main()
