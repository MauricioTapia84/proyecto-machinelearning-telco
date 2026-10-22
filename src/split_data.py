from __future__ import annotations

import csv
from statistics import median
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import deduplicate_customer_profiles
from src.processed_data import generate_processed_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
RAW_TRAIN_DIR = PROJECT_ROOT / 'data' / 'raw' / 'train'
RAW_TEST_DIR = PROJECT_ROOT / 'data' / 'raw' / 'test'
PROCESSED_TRAIN_DIR = PROJECT_ROOT / 'data' / 'processed' / 'train'
PROCESSED_TEST_DIR = PROJECT_ROOT / 'data' / 'processed' / 'test'


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open('r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        return rows, reader.fieldnames or []


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize values and remove repeated profiles, ignoring customer ID."""
    profiles = deduplicate_customer_profiles(
        pd.DataFrame(rows),
        id_column="customerID",
        total_charges_column="TotalCharges",
        tenure_column="tenure",
    )
    return profiles.astype(object).where(pd.notna(profiles), "").to_dict("records")


def split_train_test(rows: list[dict[str, str]], random_seed: int = 42) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = deduplicate_rows(rows)
    labels = [str(row.get('Churn', 'Unknown')) for row in rows]

    train_rows, test_rows, _, _ = train_test_split(
        rows,
        labels,
        test_size=0.2,
        random_state=random_seed,
        stratify=labels,
        shuffle=True,
    )

    return train_rows, test_rows


def clean_processed_rows(
    rows: list[dict[str, str]],
    fallback_median: float | None = None,
) -> list[dict[str, str]]:
    """Clean TotalCharges using only partition-appropriate information."""
    cleaned: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        total = str(item.get('TotalCharges', '')).strip()
        if total in {'', ' ', 'nan', 'NaN', 'None'}:
            item['TotalCharges'] = ''
            if str(item.get('tenure', '')).strip() == '0':
                item['TotalCharges'] = '0.0'
        cleaned.append(item)

    values = []
    for row in cleaned:
        total = str(row.get('TotalCharges', '')).strip()
        if total not in {'', ' ', 'nan', 'NaN', 'None'}:
            try:
                values.append(float(total))
            except ValueError:
                pass

    if fallback_median is None:
        fallback_median = float(median(values)) if values else 0.0

    for row in cleaned:
        total = str(row.get('TotalCharges', '')).strip()
        if total in {'', ' ', 'nan', 'NaN', 'None'}:
            row['TotalCharges'] = str(fallback_median)

    return cleaned


def match_split_rows(all_rows: list[dict[str, str]], train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_ids = {row.get('customerID') for row in train_rows}
    test_ids = {row.get('customerID') for row in test_rows}

    proc_train: list[dict[str, str]] = []
    proc_test: list[dict[str, str]] = []

    for row in all_rows:
        customer_id = row.get('customerID')
        if customer_id in train_ids:
            proc_train.append(row)
        elif customer_id in test_ids:
            proc_test.append(row)

    return proc_train, proc_test


def main() -> None:
    summary = generate_processed_datasets()
    print(f"raw_rows={summary['raw_rows']}")
    print(f"processed_train={summary['train_rows']}")
    print(f"processed_test={summary['test_rows']}")
    print(f"processed_train_columns_original={summary['train_columns_original']}")
    print(f"processed_train_columns_pca_kmeans={summary['train_columns_pca_kmeans']}")
    print('Transformadores ajustados solo con train; test solo fue transformado.')


if __name__ == '__main__':
    main()
