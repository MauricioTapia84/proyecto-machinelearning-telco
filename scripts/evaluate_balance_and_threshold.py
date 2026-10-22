"""Compare class weighting/sampling and evaluate score thresholds on OOF data."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import RANDOM_STATE, generate_oof_scores
from src.pipeline import build_processed_evaluation_pipeline
from src.processed_data import load_processed_split


OUTPUT_DIR = PROJECT_ROOT / "output" / "resultados"
RATES = (0.10, 0.20, 0.30, 0.40, 0.50)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "contacts": int(predictions.sum()),
        "contact_rate": float(predictions.mean()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def threshold_rows(
    y_train: pd.Series,
    oof_scores: np.ndarray,
    y_test: pd.Series,
    test_scores: np.ndarray,
    variant: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    default = classification_metrics(y_test, test_scores, 0.0)
    rows.append({"variant": variant, "policy": "default_0", **default})

    order = np.argsort(oof_scores)[::-1]
    for rate in RATES:
        position = max(1, int(round(rate * len(order)))) - 1
        threshold = float(oof_scores[order[position]])
        metrics = classification_metrics(y_test, test_scores, threshold)
        rows.append({
            "variant": variant,
            "policy": f"top_{int(rate * 100)}pct_oof",
            "oof_threshold": threshold,
            **metrics,
        })

    # This threshold is selected only with training OOF scores. It is reported
    # on test once, after the policy was fixed.
    oof_predictions = (oof_scores >= 0.0).astype(int)
    candidates = np.unique(oof_scores)
    f1_by_threshold = [
        (float(f1_score(y_train, oof_scores >= candidate, zero_division=0)), float(candidate))
        for candidate in candidates
    ]
    _, best_threshold = max(f1_by_threshold, key=lambda item: (item[0], -item[1]))
    metrics = classification_metrics(y_test, test_scores, best_threshold)
    rows.append({
        "variant": variant,
        "policy": "max_f1_oof",
        "oof_threshold": best_threshold,
        "oof_default_contacts": int(oof_predictions.sum()),
        **metrics,
    })
    return rows


def main() -> None:
    X_train, y_train = load_processed_split('train', representation='original')

    variants = {
        "SVM sin SMOTE + sin peso": (
            SVC(C=1.0, kernel="linear", class_weight=None, random_state=RANDOM_STATE),
            False,
        ),
        "SVM sin SMOTE + balanced": (
            SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=RANDOM_STATE),
            False,
        ),
        "SVM sin SMOTE + peso 1.5": (
            SVC(C=1.0, kernel="linear", class_weight={0: 1.0, 1: 1.5}, random_state=RANDOM_STATE),
            False,
        ),
        "SVM con SMOTE + sin peso": (
            SVC(C=1.0, kernel="linear", class_weight=None, random_state=RANDOM_STATE),
            True,
        ),
        "SVM con SMOTE + balanced": (
            SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    comparison_rows: list[dict[str, float | str | bool]] = []
    threshold_rows_out: list[dict[str, float | int | str]] = []
    for variant, (classifier, use_smote) in variants.items():
        pipeline = build_processed_evaluation_pipeline(
            classifier,
            use_smote=use_smote,
            random_state=RANDOM_STATE,
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        fitted = clone(pipeline).fit(X_train, y_train)
        X_test, y_test = load_processed_split(
            'test', representation='original',
        )
        test_scores = fitted.decision_function(X_test)
        default_metrics = classification_metrics(y_test, test_scores, 0.0)
        comparison_rows.append({
            "variant": variant,
            "use_smote": use_smote,
            "cv_recall_mean": float(scores["test_recall"].mean()),
            "cv_recall_sd": float(scores["test_recall"].std(ddof=1)),
            "cv_precision_mean": float(scores["test_precision"].mean()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_average_precision_mean": float(scores["test_average_precision"].mean()),
            "test_recall_default": default_metrics["recall"],
            "test_precision_default": default_metrics["precision"],
            "test_f1_default": default_metrics["f1"],
            "test_roc_auc": default_metrics["roc_auc"],
            "test_average_precision": default_metrics["average_precision"],
            "test_fpr_default": default_metrics["fpr"],
            "test_contacts_default": default_metrics["contacts"],
        })

        if variant == "SVM sin SMOTE + balanced":
            oof_scores = generate_oof_scores(
                X_train,
                y_train,
                classifier,
                use_smote=False,
                data_is_processed=True,
                random_state=RANDOM_STATE,
                n_splits=5,
            )
            threshold_rows_out.extend(
                threshold_rows(y_train, oof_scores, y_test, test_scores, variant)
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "evaluacion_variantes_balanceo.csv"
    threshold_path = OUTPUT_DIR / "evaluacion_umbral_sin_smote.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(threshold_rows_out).to_csv(threshold_path, index=False)
    print("=== Comparación de balanceo ===")
    print(pd.DataFrame(comparison_rows).round(4).to_string(index=False))
    print("\n=== Umbrales del SVM sin SMOTE + balanced ===")
    print(pd.DataFrame(threshold_rows_out).round(4).to_string(index=False))
    print(f"\nGuardado: {comparison_path}")
    print(f"Guardado: {threshold_path}")


if __name__ == "__main__":
    main()
