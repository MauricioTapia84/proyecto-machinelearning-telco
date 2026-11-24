"""Compare soft voting and stacking ensembles with the supervised baseline."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import CORRELATION_THRESHOLD, RANDOM_STATE, REDUCED_FEATURE_COUNT, select_reduced_features
from src.paths import RESULTS_DIR
from src.processed_data import load_processed_split


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([('classifier', classifier)])


def main() -> None:
    X_train, y_train = load_processed_split('train', representation='original')
    feature_selection = select_reduced_features(
        X_train,
        y_train,
        correlation_threshold=CORRELATION_THRESHOLD,
        n_features_to_select=REDUCED_FEATURE_COUNT,
        random_state=RANDOM_STATE,
    )
    X_train = X_train[feature_selection.selected_features]

    base_estimators = [
        ('logistic', LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', n_jobs=1, random_state=RANDOM_STATE)),
        ('gradient', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)),
        ('svm', SVC(C=0.1683454924600351, kernel='linear', class_weight='balanced', probability=True, random_state=RANDOM_STATE)),
    ]

    voting = VotingClassifier(estimators=base_estimators, voting='soft')
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE),
        stack_method='predict_proba',
        cv=5,
        n_jobs=1,
    )

    candidates = {
        'Voting soft': voting,
        'Stacking': stacking,
        'SVM ajustado (Optuna)': SVC(C=0.1683454924600351, kernel='linear', class_weight='balanced', probability=False, random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={'recall': 'recall', 'precision': 'precision', 'f1': 'f1', 'roc_auc': 'roc_auc', 'average_precision': 'average_precision'},
            n_jobs=1,
            error_score='raise',
        )
        rows.append({
            'model': name,
            'cv_recall_mean': float(scores['test_recall'].mean()),
            'cv_recall_sd': float(scores['test_recall'].std(ddof=1)),
            'cv_precision_mean': float(scores['test_precision'].mean()),
            'cv_f1_mean': float(scores['test_f1'].mean()),
            'cv_roc_auc_mean': float(scores['test_roc_auc'].mean()),
            'cv_pr_auc_mean': float(scores['test_average_precision'].mean()),
        })

    cv_table = pd.DataFrame(rows).sort_values(['cv_recall_mean', 'cv_roc_auc_mean'], ascending=False)

    X_test, y_test = load_processed_split('test', representation='original')
    X_test = X_test[feature_selection.selected_features]

    test_rows = []
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        if hasattr(pipeline, 'predict_proba'):
            scores = pipeline.predict_proba(X_test)[:, 1]
            predictions = (scores >= 0.5).astype(int)
        else:
            predictions = pipeline.predict(X_test)
            scores = pipeline.decision_function(X_test)
        test_rows.append({
            'model': name,
            'test_recall': float(recall_score(y_test, predictions, zero_division=0)),
            'test_precision': float(precision_score(y_test, predictions, zero_division=0)),
            'test_f1': float(f1_score(y_test, predictions, zero_division=0)),
            'test_roc_auc': float(roc_auc_score(y_test, scores)),
            'test_pr_auc': float(average_precision_score(y_test, scores)),
        })

    test_table = pd.DataFrame(test_rows).sort_values(['test_recall', 'test_roc_auc'], ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(RESULTS_DIR / 'comparacion_ensambles_cv.csv', index=False)
    test_table.to_csv(RESULTS_DIR / 'comparacion_ensambles_test.csv', index=False)
    print(cv_table.to_string(index=False))
    print('\nTest reservado:')
    print(test_table.to_string(index=False))


if __name__ == '__main__':
    main()
