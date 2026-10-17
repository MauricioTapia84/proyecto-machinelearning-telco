"""Training-only sensitivity checks for exact-profile deduplication."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src.data import deduplicate_customer_profiles, normalize_customer_profiles
from src.pipeline import TARGET, build_evaluation_pipeline, clean_data


def compare_deduplication_sensitivity(
    raw_data: pd.DataFrame,
    classifier: Any,
    *,
    representation: str = 'original',
    use_smote: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare CV performance on the training profiles with/without duplicates.

    A single profile-level holdout is created first. Repeated raw rows matching a
    holdout profile are excluded from both comparisons. Grouped CV keeps all
    copies of a profile in the same fold for the no-deduplication policy.
    """
    id_column = 'customerID'
    deduplicated = deduplicate_customer_profiles(
        raw_data,
        id_column=id_column,
        total_charges_column='TotalCharges',
        tenure_column='tenure',
    )
    train_profiles, _ = train_test_split(
        deduplicated,
        test_size=0.2,
        random_state=random_state,
        stratify=deduplicated[TARGET],
    )
    profile_columns = [
        column for column in deduplicated.columns if column != id_column
    ]
    train_profile_keys = set(
        pd.util.hash_pandas_object(
            train_profiles[profile_columns],
            index=False,
        ).tolist()
    )

    normalized_raw = normalize_customer_profiles(
        raw_data,
        id_column=id_column,
        total_charges_column='TotalCharges',
        tenure_column='tenure',
    )
    raw_profile_keys = pd.util.hash_pandas_object(
        normalized_raw[profile_columns],
        index=False,
    )
    raw_train = normalized_raw.loc[
        raw_profile_keys.isin(train_profile_keys)
    ].copy()
    groups = raw_profile_keys.loc[raw_train.index].to_numpy()
    raw_train = clean_data(raw_train.drop(columns=[id_column]))
    profile_groups = pd.Series(groups, index=raw_train.index)

    unique_train = raw_train.loc[
        ~profile_groups.duplicated(keep='first')
    ].copy()

    def cv_summary(
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        groups_for_cv: np.ndarray | None,
    ) -> tuple[float, float, float, float]:
        if groups_for_cv is None:
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=random_state,
            ).split(features, labels)
        else:
            cv = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=random_state,
            ).split(features, labels, groups_for_cv)
        pipeline = build_evaluation_pipeline(
            features.drop(columns=[TARGET]),
            clone(classifier),
            representation=representation,
            use_smote=use_smote,
            random_state=random_state,
        )
        scores = cross_validate(
            pipeline,
            features.drop(columns=[TARGET]),
            labels,
            cv=cv,
            scoring={'recall': 'recall', 'roc_auc': 'roc_auc'},
            return_train_score=False,
            n_jobs=1,
            error_score='raise',
        )
        return (
            float(scores['test_recall'].mean()),
            float(scores['test_recall'].std(ddof=1)),
            float(scores['test_roc_auc'].mean()),
            float(scores['test_roc_auc'].std(ddof=1)),
        )

    y_raw = raw_train[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    X_raw = raw_train.drop(columns=[TARGET])
    split_assignment = np.empty(len(raw_train), dtype=int)
    for fold, (_, validation_indices) in enumerate(
        StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        ).split(X_raw, y_raw)
    ):
        split_assignment[validation_indices] = fold
    crossed_groups = pd.DataFrame(
        {'profile': groups, 'fold': split_assignment}
    ).groupby('profile')['fold'].nunique().gt(1).sum()

    dedup_labels = unique_train[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    dedup_recall, dedup_recall_sd, dedup_auc, dedup_auc_sd = cv_summary(
        unique_train,
        dedup_labels,
        groups_for_cv=None,
    )
    raw_recall, raw_recall_sd, raw_auc, raw_auc_sd = cv_summary(
        raw_train,
        y_raw,
        groups_for_cv=groups,
    )

    return pd.DataFrame(
        [
            {
                'politica': 'Sin eliminar perfiles',
                'filas_train': len(raw_train),
                'perfiles_unicos_train': len(unique_train),
                'filas_repetidas_train': len(raw_train) - len(unique_train),
                'churn_train': float(y_raw.mean()),
                'perfiles_repartidos_entre_folds_naive': int(crossed_groups),
                'recall_cv_media': raw_recall,
                'recall_cv_sd': raw_recall_sd,
                'roc_auc_cv_media': raw_auc,
                'roc_auc_cv_sd': raw_auc_sd,
            },
            {
                'politica': 'Eliminar perfiles idénticos',
                'filas_train': len(unique_train),
                'perfiles_unicos_train': len(unique_train),
                'filas_repetidas_train': 0,
                'churn_train': float(dedup_labels.mean()),
                'perfiles_repartidos_entre_folds_naive': 0,
                'recall_cv_media': dedup_recall,
                'recall_cv_sd': dedup_recall_sd,
                'roc_auc_cv_media': dedup_auc,
                'roc_auc_cv_sd': dedup_auc_sd,
            },
        ]
    )
