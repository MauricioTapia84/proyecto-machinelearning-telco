"""Evaluate a compact feature set against the complete transformed matrix."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (
    CORRELATION_THRESHOLD,
    RANDOM_STATE,
    REDUCED_FEATURE_COUNT,
    select_model,
    select_reduced_features,
)
from src.paths import (
    CORRELATION_MATRIX_PATH,
    FEATURE_BOXPLOT_PATH,
    FEATURE_SELECTION_PATH,
    ensure_output_dirs,
)
from src.pipeline import build_processed_evaluation_pipeline
from src.processed_data import load_processed_split


def _boxplot_summary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    no_churn = X_train.loc[y_train.eq(0), features]
    churn = X_train.loc[y_train.eq(1), features]
    for feature in features:
        q1_no, q3_no = no_churn[feature].quantile([0.25, 0.75])
        q1_yes, q3_yes = churn[feature].quantile([0.25, 0.75])
        mean_no = float(no_churn[feature].mean())
        mean_yes = float(churn[feature].mean())
        rows.append({
            'feature': feature,
            'mean_no_churn': mean_no,
            'mean_churn': mean_yes,
            'mean_difference_abs': abs(mean_yes - mean_no),
            'q1_no_churn': float(q1_no),
            'q3_no_churn': float(q3_no),
            'q1_churn': float(q1_yes),
            'q3_churn': float(q3_yes),
            'mean_no_outside_churn_iqr': bool(mean_no < q1_yes or mean_no > q3_yes),
            'mean_churn_outside_no_churn_iqr': bool(mean_yes < q1_no or mean_yes > q3_no),
        })
    return pd.DataFrame(rows).sort_values(
        'mean_difference_abs', ascending=False,
    )


def _save_boxplots(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: list[str],
) -> None:
    plot_frame = X_train[features].copy()
    plot_frame['Churn'] = y_train.map({0: 'No churn', 1: 'Churn'}).to_numpy()
    long_frame = plot_frame.melt(
        id_vars='Churn',
        var_name='feature',
        value_name='value',
    )
    n_columns = 2
    n_rows = (len(features) + n_columns - 1) // n_columns
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, max(4, 3.7 * n_rows)),
        squeeze=False,
    )
    for axis, feature in zip(axes.flat, features):
        sns.boxplot(
            data=long_frame[long_frame['feature'].eq(feature)],
            x='Churn',
            y='value',
            hue='Churn',
            legend=False,
            palette='pastel',
            ax=axis,
        )
        axis.set_title(feature, fontsize=9)
        axis.set_xlabel('')
        axis.set_ylabel('Valor transformado')
    for axis in axes.flat[len(features):]:
        axis.set_visible(False)
    figure.suptitle(
        'Distribución de las variables seleccionadas según Churn\n'
        'Las cajas son un diagnóstico descriptivo; no sustituyen la validación cruzada.',
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(FEATURE_BOXPLOT_PATH, dpi=180, bbox_inches='tight')
    plt.close(figure)


def _fit_and_evaluate(
    selection,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
) -> dict[str, object]:
    train_features = X_train[features]
    test_features = X_test[features]
    pipeline = build_processed_evaluation_pipeline(
        clone(selection.selected_classifier),
        use_smote=selection.selected_smote,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(train_features, y_train)
    predictions = pipeline.predict(test_features)
    scores = pipeline.decision_function(test_features)
    return {
        'model': selection.selected_name,
        'features': len(features),
        'accuracy_test': accuracy_score(y_test, predictions),
        'recall_test': recall_score(y_test, predictions, zero_division=0),
        'precision_test': precision_score(y_test, predictions, zero_division=0),
        'f1_test': f1_score(y_test, predictions, zero_division=0),
        'roc_auc_test': roc_auc_score(y_test, scores),
        'feature_columns': ', '.join(features),
    }


def main() -> None:
    ensure_output_dirs()
    # Feature selection and both model fits use train only.
    X_train, y_train = load_processed_split('train', representation='original')
    feature_selection = select_reduced_features(
        X_train,
        y_train,
        correlation_threshold=CORRELATION_THRESHOLD,
        n_features_to_select=REDUCED_FEATURE_COUNT,
        random_state=RANDOM_STATE,
    )
    feature_selection.correlation_matrix.to_csv(CORRELATION_MATRIX_PATH)
    summary = _boxplot_summary(
        X_train,
        y_train,
        feature_selection.selected_features,
    )
    summary['selected_by_sfs'] = True
    summary['removed_by_correlation'] = False
    summary.to_csv(FEATURE_SELECTION_PATH, index=False)
    _save_boxplots(X_train, y_train, feature_selection.selected_features)

    X_train_reduced = X_train[feature_selection.selected_features]
    full_selection = select_model(
        X_train,
        y_train,
        data_is_processed=True,
        include_pca_kmeans=False,
        random_state=RANDOM_STATE,
    )
    reduced_selection = select_model(
        X_train_reduced,
        y_train,
        data_is_processed=True,
        include_pca_kmeans=False,
        random_state=RANDOM_STATE,
    )

    # Test is intentionally loaded only after selection and final train fits are ready.
    X_test, y_test = load_processed_split('test', representation='original')
    results = pd.DataFrame([
        _fit_and_evaluate(
            full_selection, X_train, y_train, X_test, y_test, list(X_train.columns),
        ),
        _fit_and_evaluate(
            reduced_selection,
            X_train_reduced,
            y_train,
            X_test,
            y_test,
            feature_selection.selected_features,
        ),
    ])
    results.to_csv(
        PROJECT_ROOT / 'output' / 'resultados' / 'comparacion_variables_reducidas.csv',
        index=False,
    )
    print('Variables eliminadas por correlación:', feature_selection.dropped_correlated)
    print('Variables seleccionadas por SFS:', feature_selection.selected_features)
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
