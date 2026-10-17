"""Group-level error metrics for a fixed binary classifier evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

_WILSON_Z_95 = 1.959963984540054


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")

    proportion = successes / total
    z_squared = _WILSON_Z_95**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        _WILSON_Z_95
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total**2)
        )
        / denominator
    )
    return float(center - margin), float(center + margin)


def calculate_group_metrics(
    y_true: Any,
    y_pred: Any,
    groups: pd.DataFrame,
    *,
    min_group_size: int = 30,
    min_class_support: int = 10,
) -> pd.DataFrame:
    """Calculate descriptive confusion and error metrics within each group.

    Small groups are retained in the output but marked as unstable. Wilson
    intervals communicate sampling uncertainty without treating differences
    between groups as proof of unfairness or causation.
    """
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("y_true y y_pred deben ser vectores de igual longitud.")
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true debe contener etiquetas binarias no vacías.")
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("y_pred debe contener únicamente 0 y 1.")
    if len(groups) != len(labels):
        raise ValueError("groups debe tener la misma cantidad de filas que y_true.")
    if groups.empty or groups.columns.has_duplicates:
        raise ValueError("groups debe incluir columnas de grupo únicas.")
    if min_group_size < 1 or min_class_support < 1:
        raise ValueError("Los mínimos de soporte deben ser enteros positivos.")

    frame = groups.reset_index(drop=True).copy()
    frame["__actual"] = labels.astype(int)
    frame["__predicted"] = predictions.astype(int)
    records: list[dict[str, Any]] = []

    for group_column in groups.columns:
        for group_value, subset in frame.groupby(
            group_column,
            dropna=False,
            observed=True,
            sort=True,
        ):
            actual = subset["__actual"].to_numpy()
            predicted = subset["__predicted"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1],
            ).ravel()
            total = len(subset)
            positive_support = int(tp + fn)
            negative_support = int(tn + fp)
            selected_count = int(tp + fp)
            recall = float(tp / positive_support) if positive_support else float("nan")
            precision = float(tp / selected_count) if selected_count else float("nan")
            false_positive_rate = (
                float(fp / negative_support) if negative_support else float("nan")
            )
            recall_low, recall_high = _wilson_interval(tp, positive_support)
            precision_low, precision_high = _wilson_interval(tp, selected_count)
            fpr_low, fpr_high = _wilson_interval(fp, negative_support)

            cautions = []
            if total < min_group_size:
                cautions.append(f"n<{min_group_size}")
            if positive_support < min_class_support:
                cautions.append(f"churn<{min_class_support}")
            if negative_support < min_class_support:
                cautions.append(f"permanencia<{min_class_support}")

            records.append({
                "variable_grupo": group_column,
                "grupo": "Sin dato" if pd.isna(group_value) else str(group_value),
                "n": total,
                "churn_real": positive_support,
                "permanencia_real": negative_support,
                "predichos_positivos": selected_count,
                "verdaderos_positivos": int(tp),
                "falsos_negativos": int(fn),
                "falsos_positivos": int(fp),
                "verdaderos_negativos": int(tn),
                "recall": recall,
                "recall_ic95_inferior": recall_low,
                "recall_ic95_superior": recall_high,
                "precision": precision,
                "precision_ic95_inferior": precision_low,
                "precision_ic95_superior": precision_high,
                "tasa_falsos_positivos": false_positive_rate,
                "fpr_ic95_inferior": fpr_low,
                "fpr_ic95_superior": fpr_high,
                "tasa_seleccion": selected_count / total,
                "advertencia_soporte": "; ".join(cautions),
            })

    return pd.DataFrame.from_records(records)
