"""Loading and prediction helpers for the persisted churn pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.paths import MODEL_PATH
from src.pipeline import TARGET, build_preprocessor, clean_data


def load_model_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    artifact = joblib.load(path)
    if (
        not isinstance(artifact, dict)
        or artifact.get('artifact_version') != 1
        or 'pipeline' not in artifact
        or 'metadata' not in artifact
    ):
        raise ValueError(f'Formato de artefacto no compatible: {path}')
    return artifact


def _raw_reference_frame() -> pd.DataFrame:
    """Return a canonical cleaned training frame that mirrors the feature schema used at fit time."""
    raw_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'train' / 'WA_Fn-UseC_-Telco-Customer-Churn_train.csv'
    if not raw_path.exists():
        raise FileNotFoundError(f'No existe el CSV de entrenamiento raw: {raw_path}')

    reference = pd.read_csv(raw_path)
    reference = clean_data(reference)
    return reference.drop(columns=[TARGET], errors='ignore')


def _prepare_input_frame(
    artifact: dict[str, Any],
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Map raw customer payloads into the feature schema expected by the trained model."""
    required_columns = artifact['metadata'].get('feature_columns')
    features = customers.copy()

    if required_columns and set(features.columns).issubset(required_columns):
        return features.reindex(columns=required_columns, fill_value=np.nan)

    raw_reference = _raw_reference_frame()
    raw_frame = clean_data(features)
    raw_frame = raw_frame.drop(columns=[TARGET], errors='ignore')
    raw_frame = raw_frame.replace({pd.NA: np.nan, None: np.nan})
    raw_frame = raw_frame.reindex(columns=raw_reference.columns, fill_value=np.nan)

    if not required_columns:
        return raw_frame

    preprocessor = build_preprocessor(raw_reference)
    preprocessor.fit(raw_reference)
    transformed = preprocessor.transform(raw_frame)
    transformed_frame = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=raw_frame.index,
    )
    return transformed_frame.reindex(columns=required_columns, fill_value=np.nan)


def score_customers(
    artifact: dict[str, Any],
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Return default-threshold predictions and ranking scores for new rows."""
    if not isinstance(customers, pd.DataFrame):
        raise TypeError('customers debe ser un DataFrame de pandas.')

    pipeline = artifact['pipeline']
    input_frame = _prepare_input_frame(artifact, customers)

    score_method = artifact['metadata'].get('score_method')
    if score_method == 'decision_function':
        try:
            scores = pipeline.decision_function(input_frame)
        except ValueError:
            if artifact['metadata'].get('feature_columns'):
                raise ValueError(
                    'Faltan predictores requeridos por el modelo o el input no tiene el formato de cliente esperado.'
                ) from None
            raise
    elif score_method == 'predict_proba':
        scores = pipeline.predict_proba(input_frame)[:, 1]
    else:
        raise ValueError(f'Método de puntuación no compatible: {score_method}')

    threshold = artifact['metadata'].get('operating_threshold')
    predictions = (
        pipeline.predict(input_frame)
        if threshold is None
        else (scores >= float(threshold)).astype(int)
    )
    return pd.DataFrame(
        {
            'prediction': predictions,
            'score': scores,
        },
        index=customers.index,
    )
