"""Loading and prediction helpers for the persisted churn pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.paths import MODEL_PATH
from src.pipeline import TARGET


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


def score_customers(
    artifact: dict[str, Any],
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Return default-threshold predictions and ranking scores for new rows."""
    if not isinstance(customers, pd.DataFrame):
        raise TypeError('customers debe ser un DataFrame de pandas.')
    features = customers.drop(columns=[TARGET], errors='ignore')
    required_columns = artifact['metadata'].get('feature_columns')
    if required_columns:
        missing_columns = [
            column for column in required_columns if column not in features.columns
        ]
        if missing_columns:
            raise ValueError(
                f'Faltan predictores requeridos por el modelo: {missing_columns}'
            )
    pipeline = artifact['pipeline']
    score_method = artifact['metadata'].get('score_method')
    if score_method == 'decision_function':
        scores = pipeline.decision_function(features)
    elif score_method == 'predict_proba':
        scores = pipeline.predict_proba(features)[:, 1]
    else:
        raise ValueError(f'Método de puntuación no compatible: {score_method}')
    threshold = artifact['metadata'].get('operating_threshold')
    predictions = (
        pipeline.predict(features)
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
