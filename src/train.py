"""Train, evaluate, and persist the canonical Telco churn pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import RANDOM_STATE, save_model_artifact, select_model
from src.paths import MODEL_PATH, RAW_DATA_PATH
from src.pipeline import TARGET, build_evaluation_pipeline, prepare_dataset


def main() -> None:
    data = prepare_dataset(pd.read_csv(RAW_DATA_PATH))
    X = data.drop(columns=[TARGET])
    y = data[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    selection = select_model(X_train, y_train, random_state=RANDOM_STATE)
    pipeline = build_evaluation_pipeline(
        X_train,
        clone(selection.selected_classifier),
        representation=selection.selected_representation,
        use_smote=selection.selected_smote,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    score_method = (
        'decision_function'
        if hasattr(pipeline, 'decision_function')
        else 'predict_proba'
    )
    scores = (
        pipeline.decision_function(X_test)
        if score_method == 'decision_function'
        else pipeline.predict_proba(X_test)[:, 1]
    )
    print(selection.cv_results.sort_values(
        ['recall_media', 'roc_auc_media'], ascending=False,
    ).round(4))
    print(f'\nModelo seleccionado por CV: {selection.selected_name}')
    print(classification_report(
        y_test,
        predictions,
        target_names=['Permanece', 'Churn'],
        zero_division=0,
    ))
    print(f'ROC-AUC de test (evaluación final): {roc_auc_score(y_test, scores):.4f}')

    metadata = {
        'model_name': selection.selected_name,
        'representation': selection.selected_representation,
        'uses_smote': selection.selected_smote,
        'target': TARGET,
        'feature_columns': list(X_train.columns),
        'score_method': score_method,
        'operating_threshold': None,
        'optuna_best_params': selection.best_params,
        'optuna_best_recall_cv': selection.optuna_best_recall,
        'optuna_best_roc_auc_cv': selection.optuna_best_roc_auc,
    }
    artifact_path = save_model_artifact(pipeline, metadata, MODEL_PATH)
    print(f'Pipeline completo guardado en: {artifact_path}')


if __name__ == '__main__':
    main()
