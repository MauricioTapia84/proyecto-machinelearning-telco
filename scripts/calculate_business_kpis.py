"""Calculate the business KPIs used in the checklist review."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model_artifact, score_customers
from src.paths import MODEL_PATH, RAW_DATA_PATH
from src.processed_data import load_processed_split


def prepare_test_data(
    representation: str,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    total_customers = len(raw_data)
    total_dataset_churn = int((raw_data["Churn"] == "Yes").sum())
    test, target = load_processed_split(
        'test', representation=representation,
    )
    return test, target, total_customers, total_dataset_churn


def main() -> None:
    artifact = load_model_artifact(MODEL_PATH)
    representation = str(artifact['metadata'].get('representation', 'original'))
    test, target, total_customers, total_dataset_churn = prepare_test_data(
        representation,
    )
    predictions = score_customers(artifact, test)

    total_churn = int(target.sum())
    true_positives = int(((predictions["prediction"] == 1) & (target == 1)).sum())
    top_count = math.ceil(len(test) * 0.20)
    ranked = pd.DataFrame(
        {"target": target.to_numpy(), "score": predictions["score"].to_numpy()}
    )
    top_risk = ranked.sort_values("score", ascending=False).head(top_count)
    captured_churn = int(top_risk["target"].sum())

    print(f"Tasa de churn dataset completo: {total_dataset_churn / total_customers:.4%}")
    print(f"Recall de churn en test: {true_positives / total_churn:.4%}")
    print(
        "Captura de churn en el 20% superior: "
        f"{captured_churn}/{total_churn} = {captured_churn / total_churn:.4%}"
    )
    print(f"Clientes test: {len(test)}; clientes priorizados: {top_count}")
    print(
        "El 20% superior es un escenario de capacidad ilustrativo; "
        "el score no es una probabilidad calibrada."
    )


if __name__ == "__main__":
    main()
