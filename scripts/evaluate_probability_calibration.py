"""Compare sigmoid and isotonic probability calibration using training-only CV."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import RANDOM_STATE, select_reduced_features
from src.processed_data import load_processed_split


def main() -> None:
    X, y = load_processed_split('train', representation='original')
    feature_selection = select_reduced_features(
        X,
        y,
        n_features_to_select=10,
        random_state=RANDOM_STATE,
    )
    X = X[feature_selection.selected_features]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []

    from sklearn.svm import SVC

    base = SVC(
        C=0.06154014789262349,
        kernel='linear',
        class_weight='balanced',
        random_state=RANDOM_STATE,
    )

    for method in ('sigmoid', 'isotonic'):
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method=method,
            cv=cv,
            ensemble=True,
        )
        probabilities = cross_val_predict(
            calibrated,
            X,
            y,
            cv=cv,
            method='predict_proba',
            n_jobs=1,
        )[:, 1]
        rows.append({
            'method': method,
            'brier_score': float(brier_score_loss(y, probabilities)),
            'roc_auc': float(roc_auc_score(y, probabilities)),
        })

    results = pd.DataFrame(rows).sort_values('brier_score')
    output = PROJECT_ROOT / 'output' / 'resultados' / 'calibracion_probabilidad.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f'Resultados guardados en: {output}')


if __name__ == '__main__':
    main()
