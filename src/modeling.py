"""Shared model selection and artifact persistence for notebook and scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.svm import SVC

from src.paths import MODEL_PATH
from src.pipeline import build_evaluation_pipeline, build_processed_evaluation_pipeline

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95
REDUCED_FEATURE_COUNT = 10
SCORING = {
    'recall': 'recall',
    'roc_auc': 'roc_auc',
    'precision': make_scorer(precision_score, zero_division=0),
    'f1': 'f1',
}


@dataclass
class ModelSelection:
    """Training-only cross-validation results and selected model recipe."""

    cv_results: pd.DataFrame
    model_specs: dict[str, tuple[Any, str, bool]]
    selected_name: str
    selected_classifier: Any
    selected_representation: str
    selected_smote: bool
    best_params: dict[str, Any]
    optuna_best_recall: float
    optuna_best_roc_auc: float


@dataclass
class FeatureSelectionResult:
    """Training-only diagnostics and selected transformed feature columns."""

    correlation_matrix: pd.DataFrame
    dropped_correlated: list[str]
    remaining_features: list[str]
    selected_features: list[str]
    selector: SequentialFeatureSelector
    estimator_name: str


def identify_highly_correlated_features(
    X_train: pd.DataFrame,
    *,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the train-only correlation matrix and redundant columns to drop."""
    if not 0 < threshold < 1:
        raise ValueError('threshold debe estar entre 0 y 1.')
    if not isinstance(X_train, pd.DataFrame) or X_train.empty:
        raise ValueError('X_train debe ser un DataFrame no vacío.')
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise TypeError('La selección requiere una matriz numérica ya transformada.')

    correlation_matrix = X_train.corr().abs()
    upper = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool),
    )
    dropped = [
        column for column in upper.columns
        if bool((upper[column] > threshold).any())
    ]
    return correlation_matrix, dropped


def select_reduced_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    n_features_to_select: int = REDUCED_FEATURE_COUNT,
    scoring: str | None = None,
    selection_estimator: str = 'linear_regression',
    random_state: int = RANDOM_STATE,
    cv_splits: int = 5,
) -> FeatureSelectionResult:
    """Filter redundant columns and select a compact subset with forward SFS.

    ``LinearRegression`` is used by default as the requested simple linear
    screening model. Since ``Churn`` is binary, it is not used as the final
    classifier; the final model remains a classifier selected separately.
    ``LogisticRegression`` is available as a classification-oriented alternative.
    """
    correlation_matrix, dropped = identify_highly_correlated_features(
        X_train,
        threshold=correlation_threshold,
    )
    reduced = X_train.drop(columns=dropped)
    if not 1 <= n_features_to_select <= reduced.shape[1]:
        raise ValueError(
            'n_features_to_select debe estar entre 1 y las columnas restantes '
            f'({reduced.shape[1]}).'
        )
    if y_train.nunique() != 2:
        raise ValueError('y_train debe contener exactamente dos clases.')

    if selection_estimator == 'linear_regression':
        estimator: Any = LinearRegression()
        selector_scoring: Any = make_scorer(
            mean_squared_error,
            greater_is_better=False,
        ) if scoring is None else scoring
        estimator_name = 'LinearRegression (filtro lineal de predictores)'
    elif selection_estimator == 'logistic_regression':
        estimator = LogisticRegression(
            C=1.0,
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
        )
        selector_scoring = 'roc_auc' if scoring is None else scoring
        estimator_name = 'LogisticRegression (modelo lineal para clasificación)'
    else:
        raise ValueError(
            'selection_estimator debe ser linear_regression o '
            'logistic_regression.'
        )
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    selector = SequentialFeatureSelector(
        estimator=estimator,
        direction='forward',
        scoring=selector_scoring,
        cv=cv,
        n_features_to_select=n_features_to_select,
        n_jobs=1,
    )
    selector.fit(reduced, y_train)
    selected_features = reduced.columns[selector.get_support()].tolist()
    return FeatureSelectionResult(
        correlation_matrix=correlation_matrix,
        dropped_correlated=dropped,
        remaining_features=reduced.columns.tolist(),
        selected_features=selected_features,
        selector=selector,
        estimator_name=estimator_name,
    )


def select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_train_pca_kmeans: pd.DataFrame | None = None,
    data_is_processed: bool = False,
    include_pca_kmeans: bool = True,
    random_state: int = RANDOM_STATE,
    n_trials: int = 5,
    timeout: int = 240,
) -> ModelSelection:
    """Compare candidate pipelines and tune an SVM using training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    model_specs: dict[str, tuple[Any, str, bool]] = {
        'Dummy (clase mayoritaria)': (
            DummyClassifier(strategy='most_frequent'), 'original', False,
        ),
        'Regresión logística': (
            LogisticRegression(
                C=1.0, solver='liblinear', max_iter=1000,
                class_weight='balanced',
                random_state=random_state,
            ),
            'original',
            False,
        ),
        'Random Forest': (
            RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight='balanced',
                random_state=random_state, n_jobs=1,
            ),
            'original',
            False,
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, random_state=random_state,
            ),
            'original',
            False,
        ),
        'SVM lineal': (
            SVC(
                C=1.0, kernel='linear', class_weight='balanced',
                random_state=random_state,
            ),
            'original',
            False,
        ),
    }
    if include_pca_kmeans:
        model_specs['SVM lineal + PCA/K-Means'] = (
            SVC(
                C=1.0, kernel='linear', class_weight='balanced',
                random_state=random_state,
            ),
            'pca_kmeans',
            False,
        )

    def features_for(representation: str) -> pd.DataFrame:
        if representation == 'pca_kmeans':
            if X_train_pca_kmeans is None:
                raise ValueError(
                    'Faltan las matrices train/X_pca_kmeans.csv para esta representación.'
                )
            return X_train_pca_kmeans
        return X_train

    def build_candidate(
        features: pd.DataFrame,
        classifier: Any,
        *,
        representation: str,
        use_smote: bool,
    ) -> Any:
        if data_is_processed:
            return build_processed_evaluation_pipeline(
                classifier,
                use_smote=use_smote,
                random_state=random_state,
            )
        return build_evaluation_pipeline(
            features,
            classifier,
            representation=representation,
            use_smote=use_smote,
            random_state=random_state,
        )

    records: dict[str, dict[str, float | str]] = {}
    for name, (classifier, representation, use_smote) in model_specs.items():
        features = features_for(representation)
        pipeline = build_candidate(
            features,
            classifier,
            representation=representation,
            use_smote=use_smote,
        )
        scores = cross_validate(
            pipeline,
            features,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=1,
            error_score='raise',
        )
        row: dict[str, float | str] = {'representacion': representation}
        for metric in SCORING:
            values = scores[f'test_{metric}']
            row[f'{metric}_media'] = float(values.mean())
            row[f'{metric}_sd'] = float(values.std(ddof=1))
        row['fit_seconds_media'] = float(scores['fit_time'].mean())
        records[name] = row

    cv_results = pd.DataFrame.from_dict(records, orient='index')
    svm_names = [name for name in ['SVM lineal', 'SVM lineal + PCA/K-Means']
                 if name in cv_results.index]
    base_svm_name = cv_results.loc[svm_names, 'recall_media'].idxmax()
    selected_representation = str(model_specs[base_svm_name][1])
    search_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=random_state,
    )
    trial_metrics: dict[int, float] = {}

    def objective(trial: optuna.Trial) -> float:
        classifier = SVC(
            C=trial.suggest_float('C', 0.03, 3.0, log=True),
            kernel='linear',
            class_weight='balanced',
            random_state=random_state,
        )
        features = features_for(selected_representation)
        candidate = build_candidate(
            features,
            classifier,
            representation=selected_representation,
            use_smote=False,
        )
        scores = cross_validate(
            candidate,
            features,
            y_train,
            cv=search_cv,
            scoring={'recall': 'recall', 'roc_auc': 'roc_auc'},
            return_train_score=False,
            n_jobs=1,
            error_score='raise',
        )
        trial_metrics[trial.number] = float(scores['test_roc_auc'].mean())
        return float(scores['test_recall'].mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    if not study.trials or study.best_trial.number not in trial_metrics:
        raise RuntimeError('Optuna no completó ninguna prueba válida.')

    best_params = dict(study.best_params)
    tuned_classifier = SVC(
        **best_params,
        kernel='linear',
        class_weight='balanced',
        random_state=random_state,
    )
    tuned_features = features_for(selected_representation)
    tuned_pipeline = build_candidate(
        tuned_features,
        tuned_classifier,
        representation=selected_representation,
        use_smote=False,
    )
    tuned_scores = cross_validate(
        tuned_pipeline,
        tuned_features,
        y_train,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
        n_jobs=1,
        error_score='raise',
    )
    tuned_row: dict[str, float | str] = {
        'representacion': selected_representation,
    }
    for metric in SCORING:
        values = tuned_scores[f'test_{metric}']
        tuned_row[f'{metric}_media'] = float(values.mean())
        tuned_row[f'{metric}_sd'] = float(values.std(ddof=1))
    tuned_row['fit_seconds_media'] = float(tuned_scores['fit_time'].mean())
    tuned_name = 'SVM ajustado (Optuna)'
    cv_results.loc[tuned_name] = tuned_row
    model_specs[tuned_name] = (tuned_classifier, selected_representation, False)

    selected_name = str(
        cv_results.sort_values(
            ['recall_media', 'roc_auc_media'], ascending=False,
        ).index[0]
    )
    classifier, representation, use_smote = model_specs[selected_name]
    return ModelSelection(
        cv_results=cv_results,
        model_specs=model_specs,
        selected_name=selected_name,
        selected_classifier=clone(classifier),
        selected_representation=representation,
        selected_smote=use_smote,
        best_params=best_params,
        optuna_best_recall=float(study.best_value),
        optuna_best_roc_auc=trial_metrics[study.best_trial.number],
    )


def generate_oof_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    classifier: Any,
    *,
    representation: str = 'original',
    use_smote: bool = True,
    data_is_processed: bool = False,
    random_state: int = RANDOM_STATE,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate training-only out-of-fold scores through the complete pipeline."""
    if n_splits < 2:
        raise ValueError('n_splits debe ser al menos 2.')

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    if data_is_processed:
        pipeline = build_processed_evaluation_pipeline(
            classifier,
            use_smote=use_smote,
            random_state=random_state,
        )
    else:
        pipeline = build_evaluation_pipeline(
            X_train,
            classifier,
            representation=representation,
            use_smote=use_smote,
            random_state=random_state,
        )
    method = (
        'decision_function'
        if hasattr(classifier, 'decision_function')
        else 'predict_proba'
    )
    scores = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method=method,
        n_jobs=1,
    )
    if scores.ndim == 2:
        if scores.shape[1] != 2:
            raise RuntimeError('Se esperaba un score binario de dos clases.')
        scores = scores[:, 1]
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise RuntimeError('La validación cruzada produjo scores no finitos.')
    return scores


def save_model_artifact(
    pipeline: Any,
    metadata: dict[str, Any],
    path: Path = MODEL_PATH,
) -> Path:
    """Persist the complete fitted pipeline and its non-secret metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            'artifact_version': 1,
            'pipeline': pipeline,
            'metadata': metadata,
        },
        path,
    )
    return path
