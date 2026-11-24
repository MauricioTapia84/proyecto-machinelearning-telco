"""Train, evaluate, and persist the canonical Telco churn pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision import (
    compute_default_operating_threshold,
    compute_quantile_operating_threshold,
)
from src.modeling import (
    CORRELATION_THRESHOLD,
    RANDOM_STATE,
    REDUCED_FEATURE_COUNT,
    generate_oof_scores,
    save_model_artifact,
    select_model,
    select_reduced_features,
)
from src.paths import MODEL_PATH
from src.pipeline import build_processed_evaluation_pipeline
from src.processed_data import load_processed_split


def main() -> None:
    # The training command intentionally loads no test file before this point.
    X_train, y_train = load_processed_split('train', representation='original')
    feature_selection = select_reduced_features(
        X_train,
        y_train,
        correlation_threshold=CORRELATION_THRESHOLD,
        n_features_to_select=REDUCED_FEATURE_COUNT,
        random_state=RANDOM_STATE,
    )
    X_train_reduced = X_train[feature_selection.selected_features]

    selection = select_model(
        X_train_reduced,
        y_train,
        data_is_processed=True,
        include_pca_kmeans=False,
        random_state=RANDOM_STATE,
    )
    selected_train = X_train_reduced
    pipeline = build_processed_evaluation_pipeline(
        clone(selection.selected_classifier),
        use_smote=selection.selected_smote,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(selected_train, y_train)

    # The test files are loaded only after model selection and final fitting.
    X_test, y_test = load_processed_split(
        'test', representation=selection.selected_representation,
    )
    X_test = X_test[selected_train.columns]
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
    oof_scores = generate_oof_scores(
        X_train_reduced,
        y_train,
        clone(selection.selected_classifier),
        representation=selection.selected_representation,
        use_smote=selection.selected_smote,
        data_is_processed=True,
        random_state=RANDOM_STATE,
    )
    raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'train' / 'WA_Fn-UseC_-Telco-Customer-Churn_train.csv'
    raw_train = pd.read_csv(raw_data_path)
    median_monthly_charge = float(
        pd.to_numeric(raw_train['MonthlyCharges'], errors='coerce').median()
    )
    operating_policy = compute_quantile_operating_threshold(
        oof_scores,
        quantile=0.75,
    )
    print(selection.cv_results.sort_values(
        ['recall_media', 'roc_auc_media'], ascending=False,
    ).round(4))
    print(f'\nModelo seleccionado por CV: {selection.selected_name}')
    print(
        'Variables seleccionadas '
        f'({len(feature_selection.selected_features)}): '
        f'{feature_selection.selected_features}'
    )
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
        'target': 'Churn',
        'input_format': 'processed_transformed_csv',
        'feature_columns': list(selected_train.columns),
        'feature_selection_method': (
            'correlation_filter_plus_forward_sequential_feature_selector'
        ),
        'feature_selection_estimator': feature_selection.estimator_name,
        'correlation_threshold': CORRELATION_THRESHOLD,
        'correlated_features_removed': feature_selection.dropped_correlated,
        'selected_feature_count': len(feature_selection.selected_features),
        'score_method': score_method,
        'probability_calibrated': False,
        'probability_method': 'sigmoid_approximation',
        'operating_threshold': float(operating_policy['threshold']),
        'operating_threshold_status': 'recalibrado OOF; validación externa pendiente',
        'operating_threshold_method': 'cuantil 75 de scores OOF para priorizar Recall',
        'test_customers': int(len(y_test)),
        'test_churn': int(y_test.sum()),
        'test_true_positive': int(((predictions == 1) & (y_test == 1)).sum()),
        'test_false_positive': int(((predictions == 1) & (y_test == 0)).sum()),
        'test_false_negative': int(((predictions == 0) & (y_test == 1)).sum()),
        'test_true_negative': int(((predictions == 0) & (y_test == 0)).sum()),
        'test_selected_count': int((scores >= float(operating_policy['threshold'])).sum()),
        'test_threshold_true_positive': int(((scores >= float(operating_policy['threshold'])) & (y_test == 1)).sum()),
        'test_threshold_false_positive': int(((scores >= float(operating_policy['threshold'])) & (y_test == 0)).sum()),
        'test_accuracy': float((predictions == y_test).mean()),
        'test_precision': float(precision_score(y_test, predictions, zero_division=0)),
        'test_recall': float(recall_score(y_test, predictions, zero_division=0)),
        'test_f1_churn': float(f1_score(y_test, predictions, zero_division=0)),
        'test_roc_auc': float(roc_auc_score(y_test, scores)),
        'test_pr_auc': float(average_precision_score(y_test, scores)),
        'operating_policy': {
            'capacity': int(operating_policy['contacts']),
            'contact_cost_ratio': 0.05,
            'contact_cost_proxy': float(median_monthly_charge * 0.05),
            'churn_loss_proxy': float(median_monthly_charge * 24),
            'retention_effectiveness': 0.20,
            'retained_value': 0.0,
        },
        'optuna_best_params': selection.best_params,
        'optuna_best_recall_cv': selection.optuna_best_recall,
        'optuna_best_roc_auc_cv': selection.optuna_best_roc_auc,
    }
    artifact_path = save_model_artifact(pipeline, metadata, MODEL_PATH)
    print(f'Pipeline completo guardado en: {artifact_path}')


if __name__ == '__main__':
    main()
