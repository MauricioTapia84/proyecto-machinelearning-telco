"""Shared deterministic normalization for customer-profile deduplication."""

from __future__ import annotations

import pandas as pd

EMPTY_VALUES = {"", "nan", "none", "null", "<na>"}


def normalize_customer_profiles(
    data: pd.DataFrame,
    *,
    id_column: str,
    total_charges_column: str,
    tenure_column: str,
) -> pd.DataFrame:
    """Normalize values relevant to profile matching without dropping any rows.

    Customer IDs are preserved but are not normalized; callers exclude them when
    constructing a profile key. The target remains part of the profile.
    """
    normalized = data.copy()
    for column in normalized.select_dtypes(include=["object", "category", "string"]):
        if column == id_column:
            continue
        normalized[column] = normalized[column].astype(object).map(
            lambda value: (
                pd.NA
                if pd.isna(value) or str(value).strip().casefold() in EMPTY_VALUES
                else str(value).strip()
            )
        )

    if total_charges_column in normalized and tenure_column in normalized:
        total_charges = pd.to_numeric(normalized[total_charges_column], errors="coerce")
        tenure = pd.to_numeric(normalized[tenure_column], errors="coerce")
        zero_tenure = total_charges.isna() & tenure.eq(0)
        normalized[total_charges_column] = total_charges.mask(zero_tenure, 0.0)

    return normalized


def deduplicate_customer_profiles(
    data: pd.DataFrame,
    *,
    id_column: str,
    total_charges_column: str,
    tenure_column: str,
) -> pd.DataFrame:
    """Normalize profile values and remove repeated profiles, ignoring customer ID.

    The target remains part of the profile when present, so otherwise identical
    observations with conflicting outcomes are retained.
    """
    normalized = normalize_customer_profiles(
        data,
        id_column=id_column,
        total_charges_column=total_charges_column,
        tenure_column=tenure_column,
    )
    profile_columns = [column for column in normalized.columns if column != id_column]
    duplicate_profiles = normalized.duplicated(subset=profile_columns, keep="first")
    return normalized.loc[~duplicate_profiles].reset_index(drop=True)
