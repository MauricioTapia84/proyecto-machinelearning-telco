"""Data preparation and model pipeline for Telco customer churn."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVC

TARGET = 'Churn'
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

        text_columns = data.select_dtypes(include=['object', 'category']).columns
        for column in text_columns:
            data[column] = data[column].replace(r'^\s*$', np.nan, regex=True)

        return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning used before model training or inference."""
    return TelcoFeatureEngineer().fit_transform(data)


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the EDA cleaning before separating features and target."""
    cleaned = clean_data(data)
    if 'TotalCharges' in cleaned:
        cleaned['TotalCharges'] = cleaned['TotalCharges'].fillna(
            cleaned['TotalCharges'].median()
        )
    return cleaned.drop_duplicates().reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-resistant transformer from the feature schema."""
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    power_columns = [
        column for column in ['tenure', 'MonthlyCharges', 'TotalCharges']
        if column in numeric_columns
    ]
    standard_columns = [column for column in ['SeniorCitizen'] if column in numeric_columns]
    remaining_numeric = [
        column for column in numeric_columns
        if column not in power_columns and column not in standard_columns
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
    if categorical_columns:
        transformers.append(
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def build_pipeline(
    X: pd.DataFrame,
    classifier: Any | None = None,
    random_state: int = 42,
) -> ImblearnPipeline:
    """Build the notebook-aligned cleaning, preprocessing, SMOTE, and SVM pipeline."""
    model = classifier or SVC(
        probability=True,
        class_weight='balanced',
        random_state=random_state,
    )
    return ImblearnPipeline([
        ('cleaner', TelcoFeatureEngineer()),
        ('preprocessor', build_preprocessor(clean_data(X))),
        ('smote', SMOTE(random_state=random_state)),
        ('classifier', model),
    ])
