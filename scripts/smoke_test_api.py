"""Run a real HTTP smoke test against the local FastAPI service."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
import uvicorn
from threading import Thread

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8011')
    args = parser.parse_args()

    if args.url == 'http://127.0.0.1:8011':
        server = Thread(
            target=uvicorn.run,
            kwargs={
                'app': 'app.main:app',
                'host': '127.0.0.1',
                'port': 8011,
                'log_level': 'warning',
            },
            daemon=True,
        )
        server.start()

        base_url = 'http://127.0.0.1:8011'
        for _ in range(30):
            try:
                response = requests.get(f'{base_url}/health', timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError('La API no respondió /health en el puerto 8011.')

        health = response.json()
        if health.get('status') != 'ok':
            raise RuntimeError(f'Health inválido: {health}')
    else:
        base_url = args.url.rstrip('/')
        for _ in range(30):
            try:
                response = requests.get(f'{base_url}/health', timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f'La API no respondió /health en {base_url}.')

        health = response.json()
        if health.get('status') != 'ok':
            raise RuntimeError(f'Health inválido: {health}')

    payload = {
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
    prediction = requests.post(f'{base_url}/predict', json=payload, timeout=5)
    if prediction.status_code != 200:
        raise RuntimeError(f'/predict respondió {prediction.status_code}: {prediction.text}')

    body = prediction.json()
    required = {
        'predicted_churn',
        'risk_score',
        'probability_churn',
        'probability_percent',
        'decision_threshold',
        'risk_label',
        'business_action',
        'metrics_context',
    }
    missing = required.difference(body)
    if missing:
        raise RuntimeError(f'/predict omite campos: {sorted(missing)}')

    print('Health:', health)
    print('Predicción:', body)


if __name__ == '__main__':
    main()
