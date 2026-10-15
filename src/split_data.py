from __future__ import annotations

import csv
from statistics import median
from pathlib import Path

from sklearn.model_selection import train_test_split

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
    """Remove exact duplicate observations while ignoring the customer ID."""
    unique_rows: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        signature = tuple(
            (key, str(value))
            for key, value in row.items()
            if key != 'customerID'
        )
        if signature not in seen:
            seen.add(signature)
            unique_rows.append(row)
    return unique_rows


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
    rows, fieldnames = load_rows(RAW_CSV)
    train_rows, test_rows = split_train_test(rows)

    write_rows(RAW_TRAIN_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn_train.csv', fieldnames, train_rows)
    write_rows(RAW_TEST_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn_test.csv', fieldnames, test_rows)

    proc_train = clean_processed_rows(train_rows)
    train_total_charges = [
        float(row['TotalCharges'])
        for row in proc_train
        if str(row.get('TotalCharges', '')).strip() not in {'', 'nan', 'NaN', 'None'}
    ]
    train_median = float(median(train_total_charges)) if train_total_charges else 0.0
    proc_test = clean_processed_rows(test_rows, fallback_median=train_median)

    processed_fieldnames = [key for key in proc_train[0].keys() if key != 'customerID']
    for row in proc_train:
        row.pop('customerID', None)
    for row in proc_test:
        row.pop('customerID', None)

    write_rows(PROCESSED_TRAIN_DIR / 'teleco_train.csv', processed_fieldnames, proc_train)
    write_rows(PROCESSED_TEST_DIR / 'teleco_test.csv', processed_fieldnames, proc_test)

    print(f'raw_train={len(train_rows)}')
    print(f'raw_test={len(test_rows)}')
    print(f'processed_train={len(proc_train)}')
    print(f'processed_test={len(proc_test)}')
    print('archivos creados en data/raw/train, data/raw/test, data/processed/train, data/processed/test')


if __name__ == '__main__':
    main()
