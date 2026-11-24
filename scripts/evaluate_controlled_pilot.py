"""Evaluate real retention and net savings from a controlled churn intervention."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import RESULTS_DIR


def wilson_difference_interval(treated: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    z = 1.959963984540054
    total = len(treated) + len(control)
    if total == 0:
        return float('nan'), float('nan')
    difference = float(np.mean(control) - np.mean(treated))
    variance = (
        np.var(control, ddof=1) / len(control)
        if len(control) > 1 else 0.0
    ) + (
        np.var(treated, ddof=1) / len(treated)
        if len(treated) > 1 else 0.0
    )
    margin = z * np.sqrt(max(variance, 0.0))
    return difference - margin, difference + margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.pilot)
    required = {
        'customer_id', 'group', 'contacted', 'retained', 'observed_churn',
        'contact_cost', 'customer_value',
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f'Faltan columnas piloto: {missing}')
    if data.empty:
        raise ValueError('El piloto no contiene observaciones.')

    for column in ('contacted', 'retained', 'observed_churn'):
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} debe contener únicamente 0 o 1.')

    for column in ('contact_cost', 'customer_value'):
        data[column] = pd.to_numeric(data[column], errors='raise')
        if (data[column] < 0).any():
            raise ValueError(f'{column} no puede contener valores negativos.')

    groups = set(data['group'].astype(str).str.lower())
    if not {'treatment', 'control'}.issubset(groups):
        raise ValueError('El piloto debe incluir group=treatment y group=control.')

    treatment = data[data['group'].str.lower() == 'treatment']
    control = data[data['group'].str.lower() == 'control']

    treatment_churn = treatment['observed_churn'].to_numpy(dtype=float)
    control_churn = control['observed_churn'].to_numpy(dtype=float)
    lower, upper = wilson_difference_interval(treatment_churn, control_churn)

    contact_cost = treatment['contact_cost'].sum()
    retained_customers = int(treatment['retained'].sum())
    retained_value = float(treatment.loc[treatment['retained'] == 1, 'customer_value'].sum())
    avoided_churn = float(control['observed_churn'].sum() - treatment['observed_churn'].sum())
    net_savings = retained_value - contact_cost
    observed_difference = float(control_churn.mean() - treatment_churn.mean())

    report = {
        'pilot_status': 'requires_approval',
        'treatment_n': len(treatment),
        'control_n': len(control),
        'treatment_churn_rate': float(treatment_churn.mean()),
        'control_churn_rate': float(control_churn.mean()),
        'observed_retention_difference': observed_difference,
        'observed_retention_difference_lower_ic95': lower,
        'observed_retention_difference_upper_ic95': upper,
        'retained_customers': retained_customers,
        'total_contact_cost': float(contact_cost),
        'retained_customer_value': retained_value,
        'avoided_churn_count': avoided_churn,
        'net_savings': net_savings,
        'threshold_status': 'provisional_pending_pilot',
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / 'piloto_controlado.csv'
    pd.DataFrame([report]).to_csv(output, index=False)
    print(pd.DataFrame([report]).to_string(index=False))
    print(f'Reporte guardado en: {output}')


if __name__ == '__main__':
    main()
