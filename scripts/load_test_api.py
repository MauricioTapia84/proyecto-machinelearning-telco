"""Lightweight concurrency smoke test for local API capacity."""

from __future__ import annotations

import argparse
import concurrent.futures
import time

import requests


PAYLOAD = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 75.5,
    'TotalCharges': 850.0,
}


def send(url: str) -> tuple[int, float]:
    started = time.perf_counter()
    response = requests.post(f'{url}/predict', json=PAYLOAD, timeout=10)
    return response.status_code, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8011')
    parser.add_argument('--requests', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda _: send(args.url.rstrip('/')), range(args.requests)))

    statuses = [status for status, _ in results]
    durations = [duration for _, duration in results]
    failures = sum(status != 200 for status in statuses)
    if failures:
        raise RuntimeError(f'Carga falló: {failures}/{args.requests} respuestas no fueron 200.')

    print(f'Solicitudes: {args.requests}; fallos: {failures}')
    print(f'Latencia media: {sum(durations) / len(durations):.2f} ms')
    print(f'Latencia máxima: {max(durations):.2f} ms')


if __name__ == '__main__':
    main()
