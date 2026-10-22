"""Compare every supervised candidate without SMOTE, with and without weighting."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import RANDOM_STATE
from src.paths import RAW_DATA_PATH
from src.pipeline import build_processed_evaluation_pipeline
from src.processed_data import load_processed_split


OUTPUT_DIR = PROJECT_ROOT / "output" / "resultados"


def get_scores(estimator: ImblearnPipeline, X: pd.DataFrame) -> np.ndarray:
    """Return a ranking score for either margins or positive-class probabilities."""
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)


def metric_values(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def fit_pipeline(
    pipeline: ImblearnPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    use_sample_weight: bool,
) -> ImblearnPipeline:
    if use_sample_weight:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        pipeline.fit(X, y, classifier__sample_weight=weights)
    else:
        pipeline.fit(X, y)
    return pipeline


def evaluate_variant(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str,
    estimator: object,
    representation: str,
    weighting: str,
    use_sample_weight: bool,
    cv: StratifiedKFold,
) -> dict[str, float | str | bool]:
    fold_metrics: list[dict[str, float]] = []
    fit_times: list[float] = []

    for train_indices, valid_indices in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_indices]
        y_fold_train = y_train.iloc[train_indices]
        X_fold_valid = X_train.iloc[valid_indices]
        y_fold_valid = y_train.iloc[valid_indices]

        pipeline = build_processed_evaluation_pipeline(
            clone(estimator),
            use_smote=False,
            random_state=RANDOM_STATE,
        )
        started = time.perf_counter()
        fit_pipeline(
            pipeline,
            X_fold_train,
            y_fold_train,
            use_sample_weight=use_sample_weight,
        )
        fit_times.append(time.perf_counter() - started)
        fold_predictions = pipeline.predict(X_fold_valid)
        fold_scores = get_scores(pipeline, X_fold_valid)
        fold_metrics.append(
            metric_values(y_fold_valid, fold_predictions, fold_scores)
        )

    final_pipeline = build_processed_evaluation_pipeline(
        clone(estimator),
        use_smote=False,
        random_state=RANDOM_STATE,
    )
    fit_pipeline(
        final_pipeline,
        X_train,
        y_train,
        use_sample_weight=use_sample_weight,
    )
    test_predictions = final_pipeline.predict(X_test)
    test_scores = get_scores(final_pipeline, X_test)
    test_metrics = metric_values(y_test, test_predictions, test_scores)

    frame = pd.DataFrame(fold_metrics)
    result: dict[str, float | str | bool] = {
        "model": model_name,
        "representation": representation,
        "weighting": weighting,
        "use_sample_weight": use_sample_weight,
        "cv_recall_mean": float(frame["recall"].mean()),
        "cv_recall_sd": float(frame["recall"].std(ddof=1)),
        "cv_precision_mean": float(frame["precision"].mean()),
        "cv_f1_mean": float(frame["f1"].mean()),
        "cv_roc_auc_mean": float(frame["roc_auc"].mean()),
        "cv_average_precision_mean": float(frame["average_precision"].mean()),
        "test_recall": test_metrics["recall"],
        "test_precision": test_metrics["precision"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_average_precision": test_metrics["average_precision"],
        "test_contacts": int(test_predictions.sum()),
        "fit_seconds_cv_mean": float(np.mean(fit_times)),
    }
    return result


def main() -> None:
    X_train, y_train = load_processed_split('train', representation='original')
    X_test, y_test = load_processed_split('test', representation='original')
    X_train_pca, y_train_pca = load_processed_split(
        'train', representation='pca_kmeans',
    )
    X_test_pca, y_test_pca = load_processed_split(
        'test', representation='pca_kmeans',
    )
    if not y_train.equals(y_train_pca) or not y_test.equals(y_test_pca):
        raise ValueError('Los targets de las representaciones procesadas no coinciden.')

    candidates: list[tuple[str, object, str, bool]] = [
        (
            "Regresión logística",
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "original",
            False,
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "original",
            False,
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
            "original",
            True,
        ),
        (
            "SVM lineal",
            SVC(C=1.0, kernel="linear", random_state=RANDOM_STATE),
            "original",
            False,
        ),
        (
            "SVM lineal + PCA/K-Means",
            SVC(C=1.0, kernel="linear", random_state=RANDOM_STATE),
            "pca_kmeans",
            False,
        ),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str | bool]] = []

    for model_name, base_estimator, representation, gradient_boosting_weights in candidates:
        candidate_X_train = (
            X_train_pca if representation == 'pca_kmeans' else X_train
        )
        candidate_X_test = (
            X_test_pca if representation == 'pca_kmeans' else X_test
        )
        rows.append(
            evaluate_variant(
                candidate_X_train,
                y_train,
                candidate_X_test,
                y_test,
                model_name=model_name,
                estimator=base_estimator,
                representation=representation,
                weighting="sin_peso",
                use_sample_weight=False,
                cv=cv,
            )
        )

        if gradient_boosting_weights:
            weighted_estimator = clone(base_estimator)
            weighting = "balanced_sample_weight"
            use_sample_weight = True
        else:
            weighted_estimator = clone(base_estimator).set_params(class_weight="balanced")
            weighting = "class_weight_balanced"
            use_sample_weight = False
        rows.append(
            evaluate_variant(
                candidate_X_train,
                y_train,
                candidate_X_test,
                y_test,
                model_name=model_name,
                estimator=weighted_estimator,
                representation=representation,
                weighting=weighting,
                use_sample_weight=use_sample_weight,
                cv=cv,
            )
        )

    output = pd.DataFrame(rows).sort_values(
        ["cv_f1_mean", "cv_roc_auc_mean"], ascending=False
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "evaluacion_todos_modelos_sin_smote.csv"
    output.to_csv(output_path, index=False)
    print("=== Todos los modelos sin SMOTE ===")
    print(output.round(4).to_string(index=False))
    print(f"\nGuardado: {output_path}")


if __name__ == "__main__":
    main()
