"""Operating-threshold selection from training-only out-of-fold predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_policy_economics(
    y_true: Any,
    y_pred: Any,
    *,
    contact_cost: float,
    churn_loss: float,
    retention_effectiveness: float,
    retained_value: float,
) -> dict[str, float | int]:
    """Break down scenario costs using explicit per-customer assumptions."""
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if (
        labels.ndim != 1
        or predictions.ndim != 1
        or len(labels) != len(predictions)
        or not len(labels)
    ):
        raise ValueError("y_true y y_pred deben ser vectores no vacíos de igual longitud.")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("y_true y y_pred deben contener etiquetas binarias.")
    if any(
        not np.isfinite(value) or value < 0
        for value in (contact_cost, churn_loss, retained_value)
    ):
        raise ValueError("Los costos y el valor retenido deben ser finitos y no negativos.")
    if not np.isfinite(retention_effectiveness) or not 0 <= retention_effectiveness <= 1:
        raise ValueError("retention_effectiveness debe estar entre 0 y 1.")

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    actual_churn = int(tp + fn)
    contacts = int(tp + fp)
    baseline_loss = actual_churn * churn_loss
    contact_cost_total = contacts * contact_cost
    avoided_loss = tp * retention_effectiveness * churn_loss
    residual_loss = baseline_loss - avoided_loss
    retained_incremental_value = tp * retention_effectiveness * retained_value
    net_cost = contact_cost_total + residual_loss - retained_incremental_value

    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "contacts": contacts,
        "baseline_loss": float(baseline_loss),
        "contact_cost_total": float(contact_cost_total),
        "gross_loss_avoided": float(avoided_loss),
        "residual_churn_loss": float(residual_loss),
        "retained_incremental_value": float(retained_incremental_value),
        "expected_net_cost": float(net_cost),
        "net_savings_vs_no_contact": float(baseline_loss - net_cost),
    }


def select_operating_threshold(
    y_true: Any,
    scores: Any,
    *,
    capacity: int,
    contact_cost: float,
    churn_loss: float,
    retention_effectiveness: float,
    retained_value: float,
) -> dict[str, float | int]:
    """Choose the minimum expected-cost score threshold under a contact cap.

    `scores` must be generated out-of-fold from training data. Costs are per
    customer, and retention effectiveness is the probability that a contacted
    churner is retained by the intervention.
    """
    labels = np.asarray(y_true, dtype=int)
    risk_scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or risk_scores.ndim != 1 or len(labels) != len(risk_scores):
        raise ValueError('y_true y scores deben ser vectores de igual longitud.')
    if not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('y_true debe contener etiquetas binarias no vacías.')
    if not np.isfinite(risk_scores).all():
        raise ValueError('scores debe contener valores finitos.')
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError('capacity debe ser un entero no negativo.')
    if capacity > len(labels):
        raise ValueError('capacity no puede superar el tamaño del conjunto.')
    if any(not np.isfinite(value) or value < 0 for value in (
        contact_cost, churn_loss, retained_value,
    )):
        raise ValueError('Los costos y el valor retenido deben ser finitos y no negativos.')
    if not np.isfinite(retention_effectiveness) or not 0 <= retention_effectiveness <= 1:
        raise ValueError('retention_effectiveness debe estar entre 0 y 1.')

    thresholds = [float('inf'), *np.unique(risk_scores).tolist()]
    candidates: list[dict[str, float | int]] = []
    total_churn = int(labels.sum())
    for threshold in thresholds:
        predictions = (risk_scores >= threshold).astype(int)
        contacts = int(predictions.sum())
        if contacts > capacity:
            continue
        economics = calculate_policy_economics(
            labels,
            predictions,
            contact_cost=contact_cost,
            churn_loss=churn_loss,
            retention_effectiveness=retention_effectiveness,
            retained_value=retained_value,
        )
        candidates.append({
            'threshold': threshold,
            'contacts': contacts,
            'true_positive': int(economics['true_positive']),
            'false_positive': int(economics['false_positive']),
            'false_negative': int(economics['false_negative']),
            'true_negative': int(economics['true_negative']),
            'recall': (
                float(economics['true_positive'] / total_churn)
                if total_churn
                else 0.0
            ),
            'precision': (
                float(economics['true_positive'] / contacts) if contacts else 0.0
            ),
            'expected_cost': float(economics['expected_net_cost']),
        })

    if not candidates:
        raise RuntimeError('No se encontró una política compatible con la capacidad.')
    return min(
        candidates,
        key=lambda row: (
            float(row['expected_cost']),
            int(row['contacts']),
            -float(row['threshold']),
        ),
    )
