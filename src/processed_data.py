"""Generate and load the canonical transformed train/test matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import deduplicate_customer_profiles, normalize_customer_profiles
from src.paths import (
    PROCESSED_TEST_DIR,
    PROCESSED_TEST_X_ORIGINAL_PATH,
    PROCESSED_TEST_X_PCA_PATH,
    PROCESSED_TEST_Y_PATH,
    PROCESSED_TRAIN_DIR,
    PROCESSED_TRAIN_X_ORIGINAL_PATH,
    PROCESSED_TRAIN_X_PCA_PATH,
    PROCESSED_TRAIN_Y_PATH,
    RAW_DATA_PATH,
)
from src.pipeline import PCAKMeansFeatures, TARGET, build_preprocessor, clean_data

Representation = Literal['original', 'pca_kmeans']
SplitName = Literal['train', 'test']


def _profile_groups(data: pd.DataFrame) -> pd.Series:
    """Return stable profile groups without using customer IDs."""
    normalized = normalize_customer_profiles(
        data,
        id_column='customerID',
        total_charges_column='TotalCharges',
        tenure_column='tenure',
    )
    profile_columns = [column for column in normalized.columns if column != 'customerID']
    return pd.util.hash_pandas_object(normalized[profile_columns], index=False)


def _split_raw_data(
    raw_data: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a profile-level holdout before fitting any transformer."""
    if TARGET not in raw_data or 'customerID' not in raw_data:
        raise ValueError('El CSV crudo debe contener customerID y Churn.')

    groups = _profile_groups(raw_data)
    labels = raw_data[TARGET].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )
    candidates = list(splitter.split(raw_data, labels, groups))
    total_churn_rate = float(labels.eq('Yes').mean())
    train_indices, test_indices = min(
        candidates,
        key=lambda split: (
            abs(len(split[1]) / len(raw_data) - 0.20),
            abs(float(labels.iloc[split[1]].eq('Yes').mean()) - total_churn_rate),
        ),
    )
    train = raw_data.iloc[train_indices].reset_index(drop=True)
    test = raw_data.iloc[test_indices].reset_index(drop=True)
    if set(groups.iloc[train_indices]).intersection(set(groups.iloc[test_indices])):
        raise RuntimeError('El split dejó perfiles idénticos en train y test.')
    return train, test


def split_raw_holdout(
    raw_data: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expose the profile-level split for EDA without fitting transforms."""
    return _split_raw_data(raw_data, random_state=random_state)


def _transform_training_and_test(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    *,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Fit every learned transform on train and only transform the holdout."""
    # Deduplication is deliberately restricted to train. Test rows remain intact.
    train_deduplicated = deduplicate_customer_profiles(
        train_raw,
        id_column='customerID',
        total_charges_column='TotalCharges',
        tenure_column='tenure',
    )
    train_clean = clean_data(train_deduplicated)
    test_clean = clean_data(test_raw)

    y_train = train_clean.pop(TARGET).map({'No': 0, 'Yes': 1}).astype(int)
    y_test = test_clean.pop(TARGET).map({'No': 0, 'Yes': 1}).astype(int)
    preprocessor = build_preprocessor(train_clean)
    train_original_array = preprocessor.fit_transform(train_clean)
    test_original_array = preprocessor.transform(test_clean)
    feature_names = preprocessor.get_feature_names_out()

    train_original = pd.DataFrame(
        train_original_array,
        columns=feature_names,
    )
    test_original = pd.DataFrame(
        test_original_array,
        columns=feature_names,
    )

    pca_kmeans = PCAKMeansFeatures(random_state=random_state)
    train_pca_array = pca_kmeans.fit_transform(train_original)
    test_pca_array = pca_kmeans.transform(test_original)
    pca_names = [
        *(f'pca_{index + 1}' for index in range(pca_kmeans.pca_.n_components_)),
        *(f'cluster_{index}' for index in range(pca_kmeans.n_clusters)),
    ]
    train_pca = pd.DataFrame(train_pca_array, columns=pca_names)
    test_pca = pd.DataFrame(test_pca_array, columns=pca_names)

    return {
        'train_original': train_original,
        'test_original': test_original,
        'train_pca_kmeans': train_pca,
        'test_pca_kmeans': test_pca,
        'train_y': y_train.to_frame(TARGET),
        'test_y': y_test.to_frame(TARGET),
    }


def generate_processed_datasets(
    raw_path: Path = RAW_DATA_PATH,
    *,
    random_state: int = 42,
) -> dict[str, int]:
    """Create transformed CSVs with a train-only fitting contract."""
    raw_data = pd.read_csv(raw_path)
    train_raw, test_raw = _split_raw_data(raw_data, random_state=random_state)
    transformed = _transform_training_and_test(
        train_raw,
        test_raw,
        random_state=random_state,
    )

    PROCESSED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    transformed['train_original'].to_csv(PROCESSED_TRAIN_X_ORIGINAL_PATH, index=False)
    transformed['test_original'].to_csv(PROCESSED_TEST_X_ORIGINAL_PATH, index=False)
    transformed['train_pca_kmeans'].to_csv(PROCESSED_TRAIN_X_PCA_PATH, index=False)
    transformed['test_pca_kmeans'].to_csv(PROCESSED_TEST_X_PCA_PATH, index=False)
    transformed['train_y'].to_csv(PROCESSED_TRAIN_Y_PATH, index=False)
    transformed['test_y'].to_csv(PROCESSED_TEST_Y_PATH, index=False)

    return {
        'raw_rows': len(raw_data),
        'train_rows': len(transformed['train_y']),
        'test_rows': len(transformed['test_y']),
        'train_columns_original': transformed['train_original'].shape[1],
        'train_columns_pca_kmeans': transformed['train_pca_kmeans'].shape[1],
    }


def _paths_for(split: SplitName, representation: Representation) -> tuple[Path, Path]:
    if split == 'train':
        x_original = PROCESSED_TRAIN_X_ORIGINAL_PATH
        x_pca = PROCESSED_TRAIN_X_PCA_PATH
        y_path = PROCESSED_TRAIN_Y_PATH
    else:
        x_original = PROCESSED_TEST_X_ORIGINAL_PATH
        x_pca = PROCESSED_TEST_X_PCA_PATH
        y_path = PROCESSED_TEST_Y_PATH
    return (x_pca if representation == 'pca_kmeans' else x_original, y_path)


def load_processed_split(
    split: SplitName,
    *,
    representation: Representation = 'original',
) -> tuple[pd.DataFrame, pd.Series]:
    """Load one transformed split; this function never loads the other split."""
    if representation not in {'original', 'pca_kmeans'}:
        raise ValueError(f'Representación no soportada: {representation}')
    x_path, y_path = _paths_for(split, representation)
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f'Faltan matrices procesadas para {split}. Ejecuta '
            'python scripts/generate_split.py antes de entrenar.'
        )
    X = pd.read_csv(x_path)
    target_frame = pd.read_csv(y_path)
    if TARGET not in target_frame:
        if 'abandono' in target_frame:
            target_frame = target_frame.rename(columns={'abandono': TARGET})
        else:
            raise ValueError(f'El archivo de target no contiene {TARGET}: {y_path}')
    y = pd.to_numeric(target_frame[TARGET], errors='raise').astype(int)
    if len(X) != len(y):
        raise ValueError(f'X e y no tienen la misma cantidad de filas en {split}.')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f'Hay valores no finitos en {x_path}.')
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f'El target procesado debe ser binario en {y_path}.')
    return X, y
