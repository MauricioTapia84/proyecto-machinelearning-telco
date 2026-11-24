"""Baseline drift report between training and incoming/test feature distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import RAW_DATA_PATH, RESULTS_DIR


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    if pd.api.types.is_numeric_dtype(reference):
        values = pd.concat([reference, current])
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(values.quantile(quantiles).to_numpy())
        if len(edges) < 3:
            return 0.0
        ref_counts = np.histogram(reference.dropna(), bins=edges)[0]
        cur_counts = np.histogram(current.dropna(), bins=edges)[0]
    else:
        categories = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
        ref_counts = np.array([(reference.dropna().astype(str) == category).sum() for category in categories])
        cur_counts = np.array([(current.dropna().astype(str) == category).sum() for category in categories])

    ref_share = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    cur_share = (cur_counts + 0.5) / (cur_counts.sum() + 0.5 * len(cur_counts))
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default=str(PROJECT_ROOT / 'data/raw/train/WA_Fn-UseC_-Telco-Customer-Churn_train.csv'))
    parser.add_argument('--current', default=str(PROJECT_ROOT / 'data/raw/test/WA_Fn-UseC_-Telco-Customer-Churn_test.csv'))
    parser.add_argument('--threshold', type=float, default=0.20)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    for dataframe in (reference, current):
        if 'TotalCharges' in dataframe:
            dataframe['TotalCharges'] = pd.to_numeric(dataframe['TotalCharges'], errors='coerce')

    rows = []
    for column in sorted(set(reference.columns) & set(current.columns)):
        if column.lower() in {'customerid', 'customer_id', 'churn'}:
            continue
        psi = population_stability_index(reference[column], current[column])
        rows.append({
            'variable': column,
            'psi': psi,
            'status': 'alerta' if psi >= args.threshold else 'estable',
        })

    report = pd.DataFrame(rows).sort_values('psi', ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / 'monitorizacion_drift.csv'
    report.to_csv(output, index=False)
    print(report.head(20).to_string(index=False))
    print(f'Reporte guardado en: {output}')


if __name__ == '__main__':
    main()
