"""Data preparation and model pipeline for Telco customer churn."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVC

from src.data import EMPTY_VALUES, deduplicate_customer_profiles

TARGET = 'Churn'


class PCAKMeansFeatures(BaseEstimator, TransformerMixin):
    """Fit PCA and K-Means on the current training fold and append cluster indicators."""

    def __init__(
        self,
        variance_threshold: float = 0.95,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> 'PCAKMeansFeatures':
        values = np.asarray(X)
        max_components = min(values.shape[0], values.shape[1])
        self.pca_ = PCA(
            n_components=min(self.variance_threshold, max_components),
            svd_solver='full',
        )
        components = self.pca_.fit_transform(values)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(components)
        return self

    def transform(self, X: Any) -> np.ndarray:
        components = self.pca_.transform(np.asarray(X))
        labels = self.kmeans_.predict(components)
        clusters = np.eye(self.n_clusters, dtype=components.dtype)[labels]
        return np.hstack([components, clusters])


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Apply the cleaning expected by the modeling notebook."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'TelcoFeatureEngineer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        data = data.drop(columns=['customerID'], errors='ignore')

        if 'TotalCharges' in data:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            zero_tenure = data['TotalCharges'].isna() & data['tenure'].eq(0)
            data.loc[zero_tenure, 'TotalCharges'] = 0.0

        text_columns = data.select_dtypes(include=['object', 'category', 'string']).columns
        for column in text_columns:
            data[column] = data[column].astype(object).map(
                lambda value: (
                    np.nan
                    if pd.isna(value) or str(value).strip().casefold() in EMPTY_VALUES
                    else str(value).strip()
                )
            )

        return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning used before model training or inference."""
    return TelcoFeatureEngineer().fit_transform(data)


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce deterministic EDA cleaning before separating features and target.

    Statistical imputation is intentionally deferred to the pipeline after
    the train/test split so that the test set cannot influence the fit.
    """
    profiles = deduplicate_customer_profiles(
        data,
        id_column="customerID",
        total_charges_column="TotalCharges",
        tenure_column="tenure",
    )
    return clean_data(profiles).reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-resistant transformer from the feature schema."""
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    # Only TotalCharges has enough positive skew to justify a power transform.
    # tenure is a count and MonthlyCharges is approximately symmetric, so both
    # are kept on their original scale and standardized for distance/margin-based
    # models. SeniorCitizen is binary categorical despite arriving as 0/1.
    power_columns = [column for column in ['TotalCharges'] if column in numeric_columns]
    standard_columns = [
        column for column in ['tenure', 'MonthlyCharges']
        if column in numeric_columns
    ]
    binary_categorical_columns = [
        column for column in ['SeniorCitizen']
        if column in numeric_columns
    ]
    remaining_numeric = [
        column for column in numeric_columns
        if (
            column not in power_columns
            and column not in standard_columns
            and column not in binary_categorical_columns
        )
    ]

    transformers = []
    if power_columns:
        transformers.append(
            ('power', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer(method='yeo-johnson')),
            ]), power_columns)
        )
    if standard_columns:
        transformers.append(
            ('standard', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), standard_columns)
        )
    if remaining_numeric:
        transformers.append(
            ('numeric', SimpleImputer(strategy='median'), remaining_numeric)
        )
    if binary_categorical_columns:
        transformers.append(
            ('binary_categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]), binary_categorical_columns)
        )
    if categorical_columns:
        transformers.append(
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        sparse_threshold=0.0,
    )


def build_evaluation_pipeline(
    X: pd.DataFrame,
    classifier: Any,
    *,
    representation: str = 'original',
    use_smote: bool = True,
    random_state: int = 42,
) -> ImblearnPipeline:
    """Build a fold-safe pipeline for model comparison on raw input features."""
    steps: list[tuple[str, Any]] = [
        ('cleaner', TelcoFeatureEngineer()),
        ('preprocessor', build_preprocessor(clean_data(X))),
    ]
    if representation == 'pca_kmeans':
        steps.append(('pca_kmeans', PCAKMeansFeatures(random_state=random_state)))
    elif representation != 'original':
        raise ValueError(f'Unsupported representation: {representation}')

    if use_smote:
        steps.append(('smote', SMOTE(random_state=random_state)))
    steps.append(('classifier', clone(classifier)))
    return ImblearnPipeline(steps)


def build_processed_evaluation_pipeline(
    classifier: Any,
    *,
    use_smote: bool = False,
    random_state: int = 42,
) -> ImblearnPipeline:
    """Build a model pipeline for matrices already transformed from train.

    The input matrices must come from ``src.processed_data``. No cleaner,
    imputer, scaler, encoder, PCA, or K-Means step is fitted here because those
    transformations were already fitted on the training partition when the
    processed CSVs were generated.
    """
    steps: list[tuple[str, Any]] = []
    if use_smote:
        steps.append(('smote', SMOTE(random_state=random_state)))
    steps.append(('classifier', clone(classifier)))
    return ImblearnPipeline(steps)


def build_pipeline(
    X: pd.DataFrame,
    classifier: Any | None = None,
    random_state: int = 42,
) -> ImblearnPipeline:
    """Build the notebook-aligned cleaning, preprocessing, SMOTE, and SVM pipeline."""
    model = classifier or SVC(
        class_weight='balanced',
        random_state=random_state,
    )
    return ImblearnPipeline([
        ('cleaner', TelcoFeatureEngineer()),
        ('preprocessor', build_preprocessor(clean_data(X))),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', model),
    ])
