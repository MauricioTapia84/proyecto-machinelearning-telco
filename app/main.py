from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model_artifact, score_customers
from src.paths import MODEL_PATH


app = FastAPI(
    title='Telco Churn Inference API',
    version='1.0.0',
    description='API local para predecir riesgo de churn usando el pipeline entrenado.',
)


class PredictionRequest(BaseModel):
    customer_id: str | None = None
    threshold: float | None = None
    model_config = ConfigDict(extra='allow')


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-float(value)))


def _label_from_probability(probability: float) -> str:
    if probability >= 0.7:
        return 'alto riesgo'
    if probability >= 0.4:
        return 'riesgo medio'
    return 'bajo riesgo'


def _build_metrics_context(metadata: dict[str, Any]) -> dict[str, float | str]:
    precision = (
        metadata.get('precision')
        or metadata.get('test_precision')
        or metadata.get('last_precision')
        or 0.0
    )
    f1_churn = (
        metadata.get('f1_churn')
        or metadata.get('test_f1_churn')
        or metadata.get('last_f1_churn')
        or 0.0
    )
    pr_auc = (
        metadata.get('pr_auc')
        or metadata.get('test_pr_auc')
        or metadata.get('cv_roc_auc_mean')
        or 0.0
    )
    return {
        'recall_focus': 'prioridad de detección de churn',
        'precision': float(precision),
        'f1_churn': float(f1_churn),
        'pr_auc': float(pr_auc),
    }


def _serialize_prediction(result: pd.DataFrame, threshold: float | None, metadata: dict[str, Any]) -> dict[str, Any]:
    raw_score = float(result['score'].iloc[0])
    score_method = metadata.get('score_method', 'decision_function')
    if score_method == 'decision_function':
        probability = _sigmoid(raw_score)
        operating_threshold = metadata.get('operating_threshold')
        decision_threshold = float(threshold) if threshold is not None else float(operating_threshold if operating_threshold is not None else 0.0)
        churn_prediction = bool(raw_score >= decision_threshold)
    elif score_method == 'predict_proba':
        probability = float(max(0.0, min(1.0, raw_score)))
        operating_threshold = metadata.get('operating_threshold')
        decision_threshold = float(threshold) if threshold is not None else float(operating_threshold if operating_threshold is not None else 0.5)
        churn_prediction = bool(probability >= decision_threshold)
    else:
        probability = _sigmoid(raw_score)
        operating_threshold = metadata.get('operating_threshold')
        decision_threshold = float(threshold) if threshold is not None else float(operating_threshold if operating_threshold is not None else 0.5)
        churn_prediction = bool(probability >= decision_threshold)

    return {
        'customer_id': result.index[0],
        'predicted_churn': churn_prediction,
        'risk_score': raw_score,
        'probability_churn': probability,
        'probability_percent': probability * 100.0,
        'decision_threshold': decision_threshold,
        'risk_label': _label_from_probability(probability),
        'business_action': 'contactar_cliente' if churn_prediction else 'mantener_en_monitorio',
        'model_name': metadata.get('model_name', 'telco_churn_pipeline'),
        'score_method': score_method,
        'metrics_context': _build_metrics_context(metadata),
    }


@app.get('/health')
def health() -> dict[str, Any]:
    try:
        artifact = load_model_artifact(MODEL_PATH)
        metadata = artifact.get('metadata', {})
        return {
            'status': 'ok',
            'artifact_version': artifact.get('artifact_version', 1),
            'model_name': metadata.get('model_name', 'telco_churn_pipeline'),
            'score_method': metadata.get('score_method', 'decision_function'),
        }
    except Exception as exc:  # pragma: no cover - defensive guard for local startup
        raise HTTPException(status_code=503, detail=f'Model artifact unavailable: {exc}') from exc


@app.post('/predict')
def predict(payload: PredictionRequest) -> dict[str, Any]:
    try:
        artifact = load_model_artifact(MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail='No existe el artefacto del modelo entrenado.') from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload_data = payload.model_dump(exclude_none=True)
    customer_id = payload_data.pop('customer_id', None)
    threshold = payload_data.pop('threshold', None)
    feature_payload = payload_data

    if 'features' in feature_payload and isinstance(feature_payload['features'], dict):
        feature_payload = feature_payload['features']
    if not feature_payload:
        raise HTTPException(status_code=400, detail='Debe enviar al menos un campo de cliente.')

    try:
        customer_frame = pd.DataFrame([feature_payload])
        scores = score_customers(artifact, customer_frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = _serialize_prediction(scores, threshold, artifact.get('metadata', {}))
    response['customer_id'] = customer_id or response.get('customer_id')
    return response


@app.get('/', response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8" />
            <title>Telco Churn Risk</title>
            <style>
                :root {
                    --bg: #f3f6fb;
                    --card: #ffffff;
                    --text: #1f2937;
                    --muted: #6b7280;
                    --primary: #2563eb;
                    --success: #1d9b63;
                    --warning: #d97706;
                    --danger: #b91c1c;
                    --border: #e5e7eb;
                }
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    background: var(--bg);
                    color: var(--text);
                }
                .wrap {
                    max-width: 1100px;
                    margin: 2rem auto;
                    padding: 0 1rem 2rem;
                }
                .card {
                    background: var(--card);
                    border-radius: 18px;
                    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
                    padding: 2rem;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 1rem;
                    margin-bottom: 1.5rem;
                }
                .tag {
                    display: inline-block;
                    padding: 0.35rem 0.7rem;
                    border-radius: 999px;
                    background: rgba(37, 99, 235, 0.08);
                    color: var(--primary);
                    font-weight: bold;
                    font-size: 0.8rem;
                }
                form {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 1rem;
                }
                label {
                    display: flex;
                    flex-direction: column;
                    gap: 0.45rem;
                    font-size: 0.88rem;
                    color: var(--muted);
                }
                input, select {
                    padding: 0.75rem 0.8rem;
                    border: 1px solid var(--border);
                    border-radius: 10px;
                    font-size: 0.95rem;
                }
                button {
                    grid-column: 1 / -1;
                    background: var(--primary);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 0.9rem 1.1rem;
                    font-weight: 700;
                    font-size: 1rem;
                    cursor: pointer;
                }
                .result {
                    margin-top: 1.5rem;
                    padding: 1.2rem;
                    border: 1px solid var(--border);
                    border-radius: 14px;
                    background: #f9fafb;
                }
                .result-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }
                .metric {
                    background: white;
                    border: 1px solid var(--border);
                    border-radius: 12px;
                    padding: 0.9rem;
                }
                .value {
                    font-size: 1.4rem;
                    font-weight: 700;
                    margin-top: 0.3rem;
                }
                .risk-high { color: var(--danger); }
                .risk-medium { color: var(--warning); }
                .risk-low { color: var(--success); }
                pre {
                    background: #0f172a;
                    color: #e2e8f0;
                    padding: 1rem;
                    border-radius: 10px;
                    overflow: auto;
                    margin-top: 1rem;
                }
            </style>
        </head>
        <body>
            <div class="wrap">
                <div class="card">
                    <div class="header">
                        <div>
                            <div class="tag">Telco Churn Risk</div>
                            <h1 style="margin-top: 0.8rem; margin-bottom: 0.3rem;">Predicción de riesgo de churn</h1>
                            <p style="margin: 0; color: var(--muted);">Formulario de validación rápida con la API local del proyecto.</p>
                        </div>
                    </div>

                    <form id="form">
                        <label>customer_id<input name="customer_id" value="CUST-001" /></label>
                        <label>gender<select name="gender"><option>Female</option><option>Male</option></select></label>
                        <label>SeniorCitizen<input type="number" name="SeniorCitizen" value="0" /></label>
                        <label>Partner<select name="Partner"><option>Yes</option><option>No</option></select></label>
                        <label>Dependents<select name="Dependents"><option>No</option><option>Yes</option></select></label>
                        <label>tenure<input type="number" name="tenure" value="12" /></label>
                        <label>PhoneService<select name="PhoneService"><option>Yes</option><option>No</option></select></label>
                        <label>MultipleLines<select name="MultipleLines"><option>No</option><option>Yes</option></select></label>
                        <label>InternetService<select name="InternetService"><option>Fiber optic</option><option>DSL</option><option>No</option></select></label>
                        <label>OnlineSecurity<select name="OnlineSecurity"><option>No</option><option>Yes</option></select></label>
                        <label>OnlineBackup<select name="OnlineBackup"><option>No</option><option>Yes</option></select></label>
                        <label>DeviceProtection<select name="DeviceProtection"><option>No</option><option>Yes</option></select></label>
                        <label>TechSupport<select name="TechSupport"><option>No</option><option>Yes</option></select></label>
                        <label>StreamingTV<select name="StreamingTV"><option>No</option><option>Yes</option></select></label>
                        <label>StreamingMovies<select name="StreamingMovies"><option>No</option><option>Yes</option></select></label>
                        <label>Contract<select name="Contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></label>
                        <label>PaperlessBilling<select name="PaperlessBilling"><option>No</option><option>Yes</option></select></label>
                        <label>PaymentMethod<select name="PaymentMethod"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer</option><option>Credit card</option></select></label>
                        <label>MonthlyCharges<input type="number" step="0.01" name="MonthlyCharges" value="75.5" /></label>
                        <label>TotalCharges<input type="number" step="0.01" name="TotalCharges" value="850.0" /></label>
                        <button type="submit">Evaluar riesgo</button>
                    </form>

                    <div id="result" class="result" aria-live="polite">
                        <strong>Resultado:</strong>
                        <div class="result-grid">
                            <div class="metric"><small>Riesgo</small><div id="riskLabel" class="value">-</div></div>
                            <div class="metric"><small>Probabilidad</small><div id="probability" class="value">-</div></div>
                            <div class="metric"><small>Acción</small><div id="action" class="value">-</div></div>
                            <div class="metric"><small>Score</small><div id="score" class="value">-</div></div>
                        </div>
                        <pre id="response">{}</pre>
                    </div>
                </div>
            </div>

            <script>
                const resultBox = document.getElementById('result');
                const riskLabel = document.getElementById('riskLabel');
                const probability = document.getElementById('probability');
                const action = document.getElementById('action');
                const score = document.getElementById('score');
                const response = document.getElementById('response');

                function renderRisk(payload) {
                    const probabilityValue = Number(payload.probability_percent || 0);
                    const risk = payload.risk_label || 'bajo riesgo';
                    const actionText = payload.business_action || 'mantener_en_monitorio';
                    riskLabel.textContent = risk;
                    riskLabel.className = 'value ' + (
                        risk.includes('alto') ? 'risk-high' : risk.includes('medio') ? 'risk-medium' : 'risk-low'
                    );
                    probability.textContent = `${probabilityValue.toFixed(1)}%`;
                    action.textContent = actionText;
                    score.textContent = Number(payload.risk_score || 0).toFixed(4);
                }

                document.getElementById('form').addEventListener('submit', async function (event) {
                    event.preventDefault();
                    const form = new FormData(event.target);
                    const payload = Object.fromEntries(form.entries());
                    payload.SeniorCitizen = Number(payload.SeniorCitizen);
                    payload.tenure = Number(payload.tenure);
                    payload.MonthlyCharges = Number(payload.MonthlyCharges);
                    payload.TotalCharges = Number(payload.TotalCharges);

                    const apiResponse = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                    const body = await apiResponse.json();
                    renderRisk(body);
                    response.textContent = JSON.stringify(body, null, 2);
                });
            </script>
        </body>
        </html>
        """
    )
