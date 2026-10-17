"""Report subgroup error metrics on the fixed final test partition."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import deduplicate_customer_profiles
from src.group_metrics import calculate_group_metrics
from src.inference import load_model_artifact
from src.modeling import RANDOM_STATE
from src.paths import GROUP_METRICS_PATH, MODEL_PATH, RAW_DATA_PATH, RESULTS_DIR
from src.pipeline import TARGET, clean_data

GROUP_COLUMNS = ("gender", "SeniorCitizen", "Contract", "InternetService")
MIN_GROUP_SIZE = 30
MIN_CLASS_SUPPORT = 10


def main() -> None:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    profiles = deduplicate_customer_profiles(
        raw_data,
        id_column="customerID",
        total_charges_column="TotalCharges",
        tenure_column="tenure",
    )
    missing_groups = set(GROUP_COLUMNS).difference(profiles.columns)
    if missing_groups:
        raise ValueError(f"Faltan columnas de grupo: {sorted(missing_groups)}")

    group_data = profiles[list(GROUP_COLUMNS)].copy()
    tenure = pd.to_numeric(profiles["tenure"], errors="raise")
    group_data["tenure_band"] = pd.cut(
        tenure,
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12 meses", "13-24 meses", "25-48 meses", "49+ meses"],
    ).astype("string")

    model_data = clean_data(profiles)
    X = model_data.drop(columns=[TARGET])
    y = model_data[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X_train, X_test, y_train, y_test, _, groups_test = train_test_split(
        X,
        y,
        group_data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    artifact = load_model_artifact(MODEL_PATH)
    metadata = artifact["metadata"]
    if list(X_train.columns) != metadata.get("feature_columns"):
        raise ValueError(
            "El esquema de datos no coincide con el artefacto; "
            "no se calcularán métricas por grupo."
        )

    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(X_test)
    results = calculate_group_metrics(
        y_test,
        predictions,
        groups_test,
        min_group_size=MIN_GROUP_SIZE,
        min_class_support=MIN_CLASS_SUPPORT,
    )
    results.insert(0, "modelo", str(metadata["model_name"]))
    results.insert(1, "criterio_clasificacion", "predict del clasificador; no es umbral operativo")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(GROUP_METRICS_PATH, index=False)
    columns = [
        "variable_grupo",
        "grupo",
        "n",
        "churn_real",
        "recall",
        "precision",
        "tasa_falsos_positivos",
        "tasa_seleccion",
        "advertencia_soporte",
    ]
    print("Métricas descriptivas por grupo en el test reservado:")
    print(results[columns].round(3).to_string(index=False))
    print(
        "\nIntervalos Wilson del 95% incluidos en el CSV. "
        "No ajustar el modelo con estas diferencias; el EDA ya exploró "
        "esta cohorte y la evaluación por grupos es post hoc."
    )
    print(f"Resultados guardados en: {GROUP_METRICS_PATH}")


if __name__ == "__main__":
    main()
