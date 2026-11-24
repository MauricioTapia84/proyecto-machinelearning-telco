from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any
import hashlib
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import pandas as pd

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
    threshold: float | None = None
    model_config = ConfigDict(extra='allow')


def _sigmoid(value: float) -> float:
    """Map an arbitrary real score to 0-1 without calibration claims."""
    value = float(value)
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _label_from_score(raw_score: float, decision_threshold: float) -> str:
    if raw_score >= decision_threshold:
        return 'alto riesgo'
    if raw_score >= 0.0:
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
    recall = (
        metadata.get('test_recall')
        or metadata.get('recall')
        or 0.0
    )
    roc_auc = (
        metadata.get('test_roc_auc')
        or metadata.get('roc_auc')
        or 0.0
    )
    return {
        'recall_focus': 'prioridad de detección de churn',
        'recall': float(recall),
        'precision': float(precision),
        'f1_churn': float(f1_churn),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'test_customers': int(metadata.get('test_customers', 1409)),
        'test_churn': int(metadata.get('test_churn', 374)),
        'true_positive': int(metadata.get('test_true_positive', 310)),
        'false_positive': int(metadata.get('test_false_positive', 302)),
        'false_negative': int(metadata.get('test_false_negative', 64)),
        'true_negative': int(metadata.get('test_true_negative', 733)),
        'operating_threshold_status': 'recalibrado con cuantil 75 de scores OOF; no aprobado; validación externa pendiente',
        'operating_policy_status': 'provisional',
        'provisional_test_selected': int(metadata.get('test_selected_count', 362)),
        'provisional_test_tp': int(metadata.get('test_threshold_true_positive', 229)),
        'provisional_test_fp': int(metadata.get('test_threshold_false_positive', 133)),
        'group_highlights': {
            'month_to_month_fp_rate': 0.801762114537445,
            'month_to_month_recall': 0.9673590504451038,
            'fiber_optic_fp_rate': 0.5685714285714286,
            'tenure_49_plus_fp_rate': 0.09947643979057591,
            'tenure_49_plus_recall': 0.41025641025641024,
        },
    }


def _serialize_prediction(result: pd.DataFrame, threshold: float | None, metadata: dict[str, Any]) -> dict[str, Any]:
    raw_score = float(result['score'].iloc[0])
    score_method = metadata.get('score_method', 'decision_function')
    if score_method == 'decision_function':
        probability = _sigmoid(raw_score)
        operating_threshold = metadata.get('operating_threshold')
        decision_threshold = float(threshold) if threshold is not None else float(operating_threshold if operating_threshold is not None else 0.0)
        churn_prediction = bool(raw_score >= decision_threshold)
        risk_label = _label_from_score(raw_score, decision_threshold)
    elif score_method == 'predict_proba':
        probability = float(max(0.0, min(1.0, raw_score)))
        operating_threshold = metadata.get('operating_threshold')
        decision_threshold = float(threshold) if threshold is not None else float(operating_threshold if operating_threshold is not None else 0.5)
        churn_prediction = bool(probability >= decision_threshold)
        risk_label = _label_from_score(probability, decision_threshold)
    else:
        probability = _sigmoid(raw_score)
        operating_threshold = metadata.get('operating_threshold')
        decision_threshold = float(threshold) if threshold is not None else float(operating_threshold if operating_threshold is not None else 0.5)
        churn_prediction = bool(probability >= decision_threshold)
        risk_label = _label_from_score(probability, decision_threshold)

    return {
        'predicted_churn': churn_prediction,
        'risk_score': raw_score,
        'probability_churn': probability,
        'probability_percent': probability * 100.0,
        'decision_threshold': decision_threshold,
        'risk_label': risk_label,
        'business_action': 'contactar_cliente' if churn_prediction else 'mantener_en_monitorio',
        'model_name': metadata.get('model_name', 'telco_churn_pipeline'),
        'score_method': score_method,
        'probability_method': (
            'calibrated'
            if metadata.get('probability_calibrated') is True
            else 'sigmoid_approximation'
        ),
        'metrics_context': _build_metrics_context(metadata),
    }


def _artifact_identity(artifact: dict[str, Any]) -> dict[str, Any]:
    metadata = artifact.get('metadata', {})
    artifact_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    return {
        'artifact_version': artifact.get('artifact_version', 1),
        'artifact_schema': artifact.get('artifact_schema', 'telco-churn-pipeline-v1'),
        'artifact_sha256': artifact_hash,
        'trained_at': metadata.get('trained_at'),
    }


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured_key = os.getenv('TELCO_API_KEY')
    if configured_key and x_api_key != configured_key:
        raise HTTPException(status_code=401, detail='API key inválida o ausente.')


@app.get('/health')
def health() -> dict[str, Any]:
    try:
        artifact = load_model_artifact(MODEL_PATH)
        metadata = artifact.get('metadata', {})
        identity = _artifact_identity(artifact)
        return {
            'status': 'ok',
            **identity,
            'model_name': metadata.get('model_name', 'telco_churn_pipeline'),
            'score_method': metadata.get('score_method', 'decision_function'),
        }
    except Exception as exc:  # pragma: no cover - defensive guard for local startup
        raise HTTPException(status_code=503, detail=f'Model artifact unavailable: {exc}') from exc


@app.post('/predict')
def predict(payload: PredictionRequest, _: None = Depends(_require_api_key)) -> dict[str, Any]:
    try:
        artifact = load_model_artifact(MODEL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail='No existe el artefacto del modelo entrenado.') from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload_data = payload.model_dump(exclude_none=True)
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
    response['artifact'] = _artifact_identity(artifact)
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
                    --bg: #eef3f9;
                    --surface: #ffffff;
                    --surface-soft: #f7f9fc;
                    --text: #152238;
                    --muted: #6b7890;
                    --primary: #1f66d1;
                    --primary-dark: #174e9f;
                    --cyan: #35b9c7;
                    --success: #1f9d74;
                    --warning: #dd9b31;
                    --danger: #d64f63;
                    --border: #dde5ef;
                }
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    min-height: 100vh;
                    font-family: Inter, "Segoe UI", Arial, sans-serif;
                    background: var(--bg);
                    color: var(--text);
                }
                .shell {
                    min-height: 100vh;
                    display: grid;
                    grid-template-columns: 168px minmax(0, 1fr);
                }
                aside {
                    background: #15304f;
                    color: white;
                    padding: 20px 12px;
                }
                .brand {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 24px;
                    font-size: 0.86rem;
                    font-weight: 800;
                    letter-spacing: 0.2px;
                }
                .brand-mark {
                    width: 32px;
                    height: 32px;
                    border-radius: 12px;
                    display: grid;
                    place-items: center;
                    background: linear-gradient(135deg, #55c7d3, #2a72c8);
                }
                .nav-item {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 10px;
                    border-radius: 10px;
                    background: rgba(255,255,255,0.10);
                    color: #f8fbff;
                    font-size: 0.92rem;
                }
                main {
                    min-width: 0;
                    padding: 22px 24px 32px;
                }
                .topbar {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    gap: 20px;
                    margin-bottom: 22px;
                }
                h1 {
                    margin: 0;
                    font-size: clamp(1.5rem, 3vw, 2.15rem);
                }
                .subtitle {
                    margin: 6px 0 0;
                    color: var(--muted);
                }
                .live {
                    padding: 8px 12px;
                    border-radius: 999px;
                    background: #e5f5ef;
                    color: #17775a;
                    font-size: 0.82rem;
                    font-weight: 800;
                }
                .layout {
                    display: grid;
                    grid-template-columns: minmax(360px, 1.05fr) minmax(420px, 1.45fr);
                    gap: 22px;
                    align-items: start;
                }
                .panel {
                    background: var(--surface);
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    box-shadow: 0 12px 34px rgba(31, 51, 79, 0.08);
                    min-width: 0;
                    overflow-wrap: anywhere;
                }
                .panel-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 18px 20px;
                    border-bottom: 1px solid var(--border);
                }
                .panel-header h2 { margin: 0; font-size: 1rem; line-height:1.35; }
                .form-body { padding: 18px 20px 20px; }
                form {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 14px;
                }
                label {
                    display: flex;
                    flex-direction: column;
                    gap: 7px;
                    color: var(--muted);
                    font-size: 0.78rem;
                    font-weight: 700;
                }
                input, select {
                    width: 100%;
                    height: 43px;
                    border: 1px solid var(--border);
                    border-radius: 9px;
                    background: white;
                    color: var(--text);
                    padding: 0 11px;
                    outline: none;
                }
                input:focus, select:focus {
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(31,102,209,0.10);
                }
                button {
                    grid-column: 1 / -1;
                    height: 48px;
                    margin-top: 4px;
                    border: 0;
                    border-radius: 10px;
                    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                    color: white;
                    font-weight: 800;
                    cursor: pointer;
                }
                button:disabled { opacity: 0.65; cursor: wait; }
                .dashboard { display: grid; gap: 16px; }
                .hero-card {
                    min-height: 218px;
                    display: grid;
                    grid-template-columns: 190px minmax(0, 1fr);
                    align-items: center;
                    padding: 18px 22px;
                }
                .gauge-wrap { display: grid; place-items: center; }
                .gauge {
                    width: 155px;
                    height: 155px;
                    border-radius: 50%;
                    display: grid;
                    place-items: center;
                    background: conic-gradient(var(--cyan) 0deg, var(--cyan) 0deg, #edf1f6 0deg);
                    transition: background 500ms ease;
                    box-shadow: inset 0 0 0 1px rgba(21,50,79,0.04);
                }
                .gauge-inner {
                    width: 112px;
                    height: 112px;
                    border-radius: 50%;
                    display: grid;
                    place-items: center;
                    text-align: center;
                    background: var(--surface);
                }
                .gauge-value { font-size: 1.7rem; font-weight: 900; }
                .gauge-label { color: var(--muted); font-size: 0.75rem; }
                .decision { min-width: 0; }
                .decision-label {
                    display: inline-block;
                    margin-bottom: 10px;
                    padding: 6px 10px;
                    border-radius: 999px;
                    background: #edf3fb;
                    color: var(--primary-dark);
                    font-size: 0.78rem;
                    font-weight: 900;
                }
                .decision h2 { margin: 0 0 8px; font-size: 1.55rem; }
                .decision p { margin: 0; color: var(--muted); line-height: 1.55; }
                .kpi-grid {
                    display: grid;
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                    gap: 14px;
                }
                .kpi {
                    min-height: 118px;
                    padding: 16px;
                    min-width: 0;
                }
                .kpi-top {
                    display:flex;
                    justify-content:space-between;
                    align-items:flex-start;
                    gap:8px;
                    color:var(--muted);
                    font-size:.78rem;
                    line-height:1.25;
                }
                .kpi-top span:last-child { text-align:right; }
                .kpi-value {
                    margin-top:10px;
                    font-size:clamp(1.1rem, 2.2vw, 1.55rem);
                    font-weight:900;
                    line-height:1.15;
                    overflow-wrap:anywhere;
                }
                .bar-track { height: 7px; margin-top: 12px; border-radius: 99px; background: #edf1f6; overflow:hidden; }
                .bar-fill { height:100%; width:0%; border-radius:99px; background:var(--primary); transition:width 500ms ease; }
                .metric-grid {
                    display: grid;
                    grid-template-columns: 1.25fr 0.75fr;
                    gap: 14px;
                }
                .metrics-panel, .method-panel { padding: 18px; }
                .metric-row { margin-top: 14px; }
                .metric-line { display:flex; justify-content:space-between; gap:10px; font-size:.82rem; line-height:1.35; }
                .metric-line strong { flex:0 0 auto; }
                .method-panel { display:grid; align-content:start; }
                .method-panel p { margin:8px 0 0; color:var(--muted); line-height:1.55; font-size:.82rem; overflow-wrap:anywhere; }
                .error { display:none; margin-top:12px; color:#a42d40; background:#fff0f2; border-radius:8px; padding:10px; font-size:.82rem; }
                .risk-high { color: var(--danger) !important; }
                .risk-medium { color: var(--warning) !important; }
                .risk-low { color: var(--success) !important; }
                .examples { padding: 14px 20px 0; }
                .example-grid {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0,1fr));
                    gap: 8px;
                }
                .example-button {
                    min-height: 72px;
                    height: auto;
                    min-width:0;
                    margin: 0;
                    border: 1px solid var(--border);
                    border-radius: 10px;
                    background: var(--surface-soft);
                    color: var(--text);
                    text-align: left;
                    padding: 9px 10px;
                    font-size: 0.78rem;
                    line-height: 1.35;
                    white-space:normal;
                    overflow-wrap:anywhere;
                    box-shadow: none;
                }
                .example-button strong { display:block; margin-bottom:4px; }
                .example-button span { color:var(--muted); font-size:.71rem; }
                .example-button.active { border-color:var(--primary); background:#edf4ff; }
                .story-steps {
                    display:grid;
                    grid-template-columns:repeat(3,minmax(0,1fr));
                    gap:10px;
                    margin-bottom:18px;
                }
                .story-step {
                    display:flex;
                    align-items:flex-start;
                    gap:9px;
                    min-height:76px;
                    padding:11px;
                    border:1px solid var(--border);
                    border-radius:12px;
                    background:var(--surface);
                }
                .story-number {
                    width:27px;
                    height:27px;
                    flex:0 0 27px;
                    display:grid;
                    place-items:center;
                    border-radius:8px;
                    background:#eaf2fc;
                    color:var(--primary-dark);
                    font-weight:900;
                    font-size:.78rem;
                }
                .story-step strong { display:block; font-size:.82rem; line-height:1.3; }
                .story-step span { display:block; margin-top:3px; color:var(--muted); font-size:.72rem; line-height:1.45; }
                .guide-copy {
                    margin:10px 0 0;
                    border-left:3px solid var(--cyan);
                    padding:8px 10px;
                    background:#f3fafb;
                    color:#526178;
                    border-radius:6px;
                    font-size:.78rem;
                    line-height:1.45;
                }
                @media (max-width: 1180px) {
                    .layout { grid-template-columns: minmax(320px, 0.95fr) minmax(380px, 1.05fr); }
                    .example-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
                }
                @media (max-width: 1050px) {
                    .layout { grid-template-columns: 1fr; }
                    .shell { grid-template-columns:68px minmax(0,1fr); }
                    aside { padding:14px 8px; }
                    .brand { justify-content:center; margin-bottom:18px; }
                    .brand span:last-child { display:none; }
                    .nav-item { justify-content:center; padding:10px 6px; }
                    .nav-item span:last-child { display:none; }
                    main { padding:20px 18px 28px; }
                    .metric-grid { grid-template-columns:1fr 0.75fr; }
                }
                @media (max-width: 760px) {
                    .shell { display:block; }
                    aside {
                        height:56px;
                        display:flex;
                        align-items:center;
                        padding:8px 14px;
                    }
                    .brand { margin:0; }
                    .brand span:last-child { display:inline; }
                    .nav-item { display:none; }
                    main { padding:16px 12px 24px; }
                    .topbar { align-items:flex-start; }
                    .live { display:none; }
                    .story-steps { grid-template-columns:1fr; }
                    .layout { gap:14px; }
                    form { grid-template-columns:1fr 1fr; }
                    .kpi-grid { grid-template-columns:1fr 1fr; }
                    .metric-grid { grid-template-columns:1fr; }
                    .hero-card { grid-template-columns:145px minmax(0,1fr); }
                    .example-grid { grid-template-columns:repeat(2, minmax(0,1fr)); }
                    .panel-header { padding:16px; }
                    .form-body { padding:16px; }
                }
                @media (max-width: 520px) {
                    main { padding:12px 8px 18px; }
                    h1 { font-size:1.28rem; }
                    .subtitle { font-size:.82rem; line-height:1.4; }
                    .panel { border-radius:14px; }
                    .panel-header { align-items:flex-start; gap:8px; }
                    form { grid-template-columns:1fr; }
                    .kpi-grid { grid-template-columns:1fr; }
                    .metric-grid { grid-template-columns:1fr; }
                    .hero-card { grid-template-columns:1fr; text-align:center; padding:16px 12px; }
                    .gauge { width:128px; height:128px; }
                    .gauge-inner { width:92px; height:92px; }
                    .decision h2 { font-size:1.18rem; }
                    .decision p, .guide-copy { text-align:left; }
                    .example-grid { grid-template-columns:1fr; }
                    .example-button { min-height:58px; }
                    .story-step { min-height:62px; }
                    .kpi { min-height:108px; }
                }
                .results-window { margin-top:16px; }
                .confusion-grid {
                    display:grid;
                    grid-template-columns:repeat(4,minmax(0,1fr));
                    gap:8px;
                }
                .confusion-cell {
                    min-width:0;
                    padding:12px;
                    border-radius:10px;
                    background:var(--surface-soft);
                    border:1px solid var(--border);
                }
                .confusion-cell strong { display:block; font-size:1.15rem; }
                .confusion-cell span { display:block; margin-top:4px; color:var(--muted); font-size:.72rem; line-height:1.35; }
                .subgroup-list { display:grid; gap:10px; }
                .subgroup-row { display:grid; gap:5px; }
                .subgroup-line { display:flex; justify-content:space-between; gap:8px; font-size:.78rem; }
                @media (max-width:520px) { .confusion-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
            </style>
        </head>
        <body>
            <div class="shell">
                <aside>
                    <div class="brand"><div class="brand-mark">TC</div><span>Telco Analytics</span></div>
                    <div class="nav-item"><span>◆</span><span>Panel de abandono</span></div>
                </aside>

                <main>
                    <div class="topbar">
                        <div>
                            <h1>Evaluación de abandono</h1>
                            <p class="subtitle">Modelo de priorización basado en el perfil del cliente</p>
                        </div>
                        <div class="live">● Modelo activo</div>
                    </div>

                    <div class="story-steps">
                        <div class="story-step"><div class="story-number">1</div><div><strong>Describe el perfil</strong><span>Elige un ejemplo o completa los datos disponibles del cliente.</span></div></div>
                        <div class="story-step"><div class="story-number">2</div><div><strong>Interpreta el resultado</strong><span>Compara el score con el umbral operativo y revisa el nivel de riesgo.</span></div></div>
                        <div class="story-step"><div class="story-number">3</div><div><strong>Decide la siguiente acción</strong><span>Contactar, monitorear o revisar requiere criterio humano y contexto.</span></div></div>
                    </div>

                    <div class="layout">
                        <section class="panel">
                            <div class="panel-header">
                                <h2>Perfil del cliente</h2>
                                <span style="color:var(--muted); font-size:.78rem">Sin identificador</span>
                            </div>
                            <div class="form-body">
                                <div class="examples">
                                    <div class="example-grid">
                                        <button type="button" class="example-button" data-example="stable"><strong>Perfil estable</strong><span>Contrato largo, antigüedad alta</span></button>
                                        <button type="button" class="example-button" data-example="at-risk"><strong>Perfil vulnerable</strong><span>Mes a mes, fibra y poca antigüedad</span></button>
                                        <button type="button" class="example-button" data-example="new"><strong>Perfil inicial</strong><span>Antigüedad mínima y cargos altos</span></button>
                                        <button type="button" class="example-button" data-example="premium"><strong>Perfil premium</strong><span>Servicios completos y contrato largo</span></button>
                                    </div>
                                </div>
                                <form id="form">
                                    <label>Género<select name="gender"><option value="Female">Femenino</option><option value="Male">Masculino</option></select></label>
                                    <label>Adulto mayor<select name="SeniorCitizen"><option value="0">No</option><option value="1">Sí</option></select></label>
                                    <label>Socio o pareja<select name="Partner"><option value="Yes">Sí</option><option value="No">No</option></select></label>
                                    <label>Tiene dependientes<select name="Dependents"><option value="No">No</option><option value="Yes">Sí</option></select></label>
                                    <label>Antigüedad (meses)<input type="number" min="0" name="tenure" value="12" /></label>
                                    <label>Servicio telefónico<select name="PhoneService"><option value="Yes">Sí</option><option value="No">No</option></select></label>
                                    <label>Líneas múltiples<select name="MultipleLines"><option value="No">No</option><option value="Yes">Sí</option><option value="No phone service">Sin servicio telefónico</option></select></label>
                                    <label>Servicio de internet<select name="InternetService"><option value="Fiber optic">Fibra óptica</option><option value="DSL">DSL</option><option value="No">Sin internet</option></select></label>
                                    <label>Seguridad en línea<select name="OnlineSecurity"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin servicio de internet</option></select></label>
                                    <label>Respaldo en línea<select name="OnlineBackup"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin servicio de internet</option></select></label>
                                    <label>Protección de dispositivo<select name="DeviceProtection"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin servicio de internet</option></select></label>
                                    <label>Soporte técnico<select name="TechSupport"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin servicio de internet</option></select></label>
                                    <label>Televisión por internet<select name="StreamingTV"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin servicio de internet</option></select></label>
                                    <label>Películas por internet<select name="StreamingMovies"><option value="No">No</option><option value="Yes">Sí</option><option value="No internet service">Sin servicio de internet</option></select></label>
                                    <label>Tipo de contrato<select name="Contract"><option value="Month-to-month">Mes a mes</option><option value="One year">Un año</option><option value="Two year">Dos años</option></select></label>
                                    <label>Facturación digital<select name="PaperlessBilling"><option value="No">No</option><option value="Yes">Sí</option></select></label>
                                    <label>Método de pago<select name="PaymentMethod"><option value="Electronic check">Cheque electrónico</option><option value="Mailed check">Cheque por correo</option><option value="Bank transfer (automatic)">Transferencia automática</option><option value="Credit card (automatic)">Tarjeta automática</option></select></label>
                                    <label>Cargos mensuales<input type="number" step="0.01" min="0" name="MonthlyCharges" value="75.5" /></label>
                                    <label>Cargos acumulados<input type="number" step="0.01" min="0" name="TotalCharges" value="850.0" /></label>
                                    <button id="submitButton" type="submit">Analizar cliente</button>
                                </form>
                                <div id="error" class="error"></div>
                            </div>
                        </section>

                        <section class="dashboard" aria-live="polite">
                            <div class="panel hero-card">
                                <div class="gauge-wrap">
                                    <div id="gauge" class="gauge">
                                        <div class="gauge-inner">
                                            <div><div id="probabilityValue" class="gauge-value">—</div><div class="gauge-label">índice aproximado</div></div>
                                        </div>
                                    </div>
                                </div>
                                <div class="decision">
                                    <span id="riskBadge" class="decision-label">Esperando análisis</span>
                                    <h2 id="riskTitle">Completa el perfil</h2>
                                    <p id="actionText">El resultado mostrará la decisión operativa y la acción recomendada.</p>
                                    <div id="adminGuide" class="guide-copy">Esta herramienta sirve para ordenar revisiones. No confirma que el cliente abandonará ni reemplaza la decisión humana.</div>
                                </div>
                            </div>

                            <div class="kpi-grid">
                                <div class="panel kpi">
                                    <div class="kpi-top"><span>Indicador de riesgo</span><span>Nivel calculado</span></div>
                                    <div id="scoreValue" class="kpi-value">—</div>
                                    <div class="bar-track"><div id="scoreBar" class="bar-fill"></div></div>
                                </div>
                                <div class="panel kpi">
                                    <div class="kpi-top"><span>Umbral operativo</span><span>Referencia histórica</span></div>
                                    <div id="thresholdValue" class="kpi-value">—</div>
                                    <div class="bar-track"><div id="thresholdBar" class="bar-fill"></div></div>
                                </div>
                                <div class="panel kpi">
                                    <div class="kpi-top"><span>Recomendación</span><span>Resultado del análisis</span></div>
                                    <div id="decisionValue" class="kpi-value">—</div>
                                    <div class="bar-track"><div id="decisionBar" class="bar-fill"></div></div>
                                </div>
                            </div>

                            <div class="metric-grid">
                                <div class="panel metrics-panel">
                                    <div class="panel-header" style="padding:0 0 14px">
                                        <h2>Desempeño histórico del modelo</h2>
                                    </div>
                                    <div class="metric-row"><div class="metric-line"><span>Clientes con abandono detectados</span><strong id="metricRecall">—</strong></div><div class="bar-track"><div id="recallBar" class="bar-fill"></div></div></div>
                                    <div class="metric-row"><div class="metric-line"><span>Señalamientos que fueron correctos</span><strong id="metricPrecision">—</strong></div><div class="bar-track"><div id="precisionBar" class="bar-fill"></div></div></div>
                                    <div class="metric-row"><div class="metric-line"><span>Equilibrio entre aciertos y alertas</span><strong id="metricF1">—</strong></div><div class="bar-track"><div id="f1Bar" class="bar-fill"></div></div></div>
                                    <div class="metric-row"><div class="metric-line"><span>Capacidad de separar ambos perfiles</span><strong id="metricAuc">—</strong></div><div class="bar-track"><div id="aucBar" class="bar-fill"></div></div></div>
                                </div>
                                <div class="panel method-panel">
                                    <h2>Método</h2>
                                    <p><strong>Administrativo:</strong> si el perfil cruza el umbral, se recomienda revisar una posible acción de retención.</p>
                                    <p><strong>Técnico:</strong> el score es `decision_function`; el porcentaje es una aproximación sigmoid y no una probabilidad calibrada.</p>
                                    <p id="methodText">La decisión final debe validar capacidad, costo y supervisión humana.</p>
                                </div>
                            </div>
                        </section>

                        <section class="panel results-window">
                            <div class="panel-header" style="padding:0 0 14px">
                                <h2>Resultados verificados del modelo</h2>
                                <span style="color:var(--muted);font-size:.72rem">Test reservado: 1.409 clientes</span>
                            </div>
                            <div class="confusion-grid">
                                <div class="confusion-cell"><strong id="countTP">310</strong><span>Verdaderos positivos<br>abandono detectado</span></div>
                                <div class="confusion-cell"><strong id="countFP">302</strong><span>Falsos positivos<br>permanencia señalada</span></div>
                                <div class="confusion-cell"><strong id="countFN">64</strong><span>Falsos negativos<br>abandono omitido</span></div>
                                <div class="confusion-cell"><strong id="countTN">733</strong><span>Verdaderos negativos<br>permanencia detectada</span></div>
                            </div>
                            <div class="subgroup-list" style="margin-top:16px">
                                <div class="subgroup-row">
                                    <div class="subgroup-line"><span>Contrato mes a mes: falsos positivos</span><strong>80.2%</strong></div>
                                    <div class="bar-track"><div class="bar-fill" style="width:80.2%"></div></div>
                                </div>
                                <div class="subgroup-row">
                                    <div class="subgroup-line"><span>Fibra óptica: falsos positivos</span><strong>56.9%</strong></div>
                                    <div class="bar-track"><div class="bar-fill" style="width:56.9%"></div></div>
                                </div>
                                <div class="subgroup-row">
                                    <div class="subgroup-line"><span>Antigüedad 49+ meses: falsos positivos</span><strong>9.9%</strong></div>
                                    <div class="bar-track"><div class="bar-fill" style="width:9.9%;background:var(--success)"></div></div>
                                </div>
                            </div>
                            <p class="guide-copy">El modelo detectó 310 de 374 abandonos reales al umbral por defecto del clasificador, con 302 falsas alertas sobre permanencia. El cuantil 75 de scores OOF produjo 362 seleccionados en test, con 229 TP y 133 FP. Sigue siendo provisional y requiere validación externa.</p>
                        </section>
                    </div>
                </main>
            </div>

            <script>
                const gauge = document.getElementById('gauge');
                const probabilityValue = document.getElementById('probabilityValue');
                const riskBadge = document.getElementById('riskBadge');
                const riskTitle = document.getElementById('riskTitle');
                const actionText = document.getElementById('actionText');
                const scoreValue = document.getElementById('scoreValue');
                const thresholdValue = document.getElementById('thresholdValue');
                const decisionValue = document.getElementById('decisionValue');
                const submitButton = document.getElementById('submitButton');
                const errorBox = document.getElementById('error');

                function setBar(id, value) {
                    document.getElementById(id).style.width = `${Math.max(0, Math.min(100, value))}%`;
                }

                function renderRisk(payload) {
                    const percent = Number(payload.probability_percent || 0);
                    const score = Number(payload.risk_score || 0);
                    const threshold = Number(payload.decision_threshold || 0);
                    const risk = payload.risk_label || 'bajo riesgo';
                    const predicted = Boolean(payload.predicted_churn);
                    const action = payload.business_action || 'mantener_en_monitorio';
                    const metrics = payload.metrics_context || {};

                    probabilityValue.textContent = `${percent.toFixed(1)}%`;
                    const gaugeColor = risk.includes('alto') ? '#d64f63' : risk.includes('medio') ? '#dd9b31' : '#1f9d74';
                    gauge.style.background = `conic-gradient(${gaugeColor} 0deg ${percent * 3.6}deg, #edf1f6 ${percent * 3.6}deg 360deg)`;

                    riskBadge.textContent = risk;
                    riskBadge.className = 'decision-label ' + (risk.includes('alto') ? 'risk-high' : risk.includes('medio') ? 'risk-medium' : 'risk-low');
                    riskTitle.textContent = predicted ? 'Revisar la prioridad de abandono' : 'Mantener bajo seguimiento';
                    actionText.textContent = action === 'contactar_cliente'
                        ? 'El perfil supera el umbral operativo y requiere una revisión de retención.'
                        : 'El perfil no supera el umbral operativo; mantener monitorio y supervisión.';

                    scoreValue.textContent = score.toFixed(4);
                    thresholdValue.textContent = threshold.toFixed(4);
                    decisionValue.textContent = predicted ? 'Priorizar revisión' : 'Mantener seguimiento';
                    decisionValue.className = 'kpi-value ' + (predicted ? 'risk-high' : 'risk-low');

                    setBar('scoreBar', Math.max(0, Math.min(100, (score - threshold + 2) * 25)));
                    setBar('thresholdBar', 50);
                    setBar('decisionBar', predicted ? 100 : 18);

                    const recall = Number(metrics.recall || 0) * 100;
                    const precision = Number(metrics.precision || 0) * 100;
                    const f1 = Number(metrics.f1_churn || 0) * 100;
                    const auc = Number(metrics.roc_auc || 0) * 100;
                    document.getElementById('metricRecall').textContent = `${recall.toFixed(1)}%`;
                    document.getElementById('metricPrecision').textContent = `${precision.toFixed(1)}%`;
                    document.getElementById('metricF1').textContent = `${f1.toFixed(1)}%`;
                    document.getElementById('metricAuc').textContent = `${auc.toFixed(1)}%`;
                    setBar('recallBar', recall);
                    setBar('precisionBar', precision);
                    setBar('f1Bar', f1);
                    setBar('aucBar', auc);

                    document.getElementById('adminGuide').textContent = predicted
                        ? 'El score supera el umbral operativo: considere una revisión preventiva, no una decisión automática.'
                        : 'El score queda debajo del umbral operativo: mantenga monitorio y revise cambios futuros del perfil.';
                    document.getElementById('methodText').textContent = payload.probability_method === 'sigmoid_approximation'
                        ? 'Técnicamente, esta salida es una aproximación sigmoid del decision_function y no una probabilidad calibrada. Administrativamente, úsela para priorizar revisión.'
                        : 'Técnicamente, la salida porcentual proviene del método definido por el artefacto. Administrativamente, úsela junto con capacidad y costo.';

                    document.getElementById('countTP').textContent = String(metrics.true_positive || 310);
                    document.getElementById('countFP').textContent = String(metrics.false_positive || 302);
                    document.getElementById('countFN').textContent = String(metrics.false_negative || 64);
                    document.getElementById('countTN').textContent = String(metrics.true_negative || 733);
                }

                document.getElementById('form').addEventListener('submit', async function(event) {
                    event.preventDefault();
                    submitButton.disabled = true;
                    submitButton.textContent = 'Analizando perfil...';
                    errorBox.style.display = 'none';
                    const form = new FormData(event.target);
                    const payload = Object.fromEntries(form.entries());
                    payload.SeniorCitizen = Number(payload.SeniorCitizen);
                    payload.tenure = Number(payload.tenure);
                    payload.MonthlyCharges = Number(payload.MonthlyCharges);
                    payload.TotalCharges = Number(payload.TotalCharges);

                    try {
                        const apiResponse = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload),
                        });
                        const body = await apiResponse.json();
                        if (!apiResponse.ok) throw new Error(body.detail || 'No fue posible evaluar el perfil');
                        renderRisk(body);
                    } catch (error) {
                        errorBox.textContent = error.message;
                        errorBox.style.display = 'block';
                    } finally {
                        submitButton.disabled = false;
                        submitButton.textContent = 'Analizar cliente';
                    }
                });

                const examples = {
                    stable: {
                        gender:'Male', SeniorCitizen:0, Partner:'Yes', Dependents:'Yes', tenure:72,
                        PhoneService:'Yes', MultipleLines:'Yes', InternetService:'DSL', OnlineSecurity:'Yes',
                        OnlineBackup:'Yes', DeviceProtection:'Yes', TechSupport:'Yes', StreamingTV:'No',
                        StreamingMovies:'No', Contract:'Two year', PaperlessBilling:'No',
                        PaymentMethod:'Credit card (automatic)', MonthlyCharges:55.2, TotalCharges:3980.0
                    },
                    'at-risk': {
                        gender:'Female', SeniorCitizen:0, Partner:'No', Dependents:'No', tenure:3,
                        PhoneService:'Yes', MultipleLines:'No', InternetService:'Fiber optic', OnlineSecurity:'No',
                        OnlineBackup:'No', DeviceProtection:'No', TechSupport:'No', StreamingTV:'Yes',
                        StreamingMovies:'Yes', Contract:'Month-to-month', PaperlessBilling:'Yes',
                        PaymentMethod:'Electronic check', MonthlyCharges:96.4, TotalCharges:288.0
                    },
                    new: {
                        gender:'Male', SeniorCitizen:1, Partner:'No', Dependents:'No', tenure:1,
                        PhoneService:'Yes', MultipleLines:'No', InternetService:'Fiber optic', OnlineSecurity:'No',
                        OnlineBackup:'Yes', DeviceProtection:'No', TechSupport:'No', StreamingTV:'Yes',
                        StreamingMovies:'Yes', Contract:'Month-to-month', PaperlessBilling:'Yes',
                        PaymentMethod:'Electronic check', MonthlyCharges:91.8, TotalCharges:91.8
                    },
                    premium: {
                        gender:'Female', SeniorCitizen:0, Partner:'Yes', Dependents:'Yes', tenure:60,
                        PhoneService:'Yes', MultipleLines:'Yes', InternetService:'Fiber optic', OnlineSecurity:'Yes',
                        OnlineBackup:'Yes', DeviceProtection:'Yes', TechSupport:'Yes', StreamingTV:'Yes',
                        StreamingMovies:'Yes', Contract:'Two year', PaperlessBilling:'No',
                        PaymentMethod:'Bank transfer (automatic)', MonthlyCharges:118.5, TotalCharges:7100.0
                    }
                };

                function fillExample(profileName) {
                    const profile = examples[profileName];
                    if (!profile) return;
                    document.querySelectorAll('.example-button').forEach(button => {
                        button.classList.toggle('active', button.dataset.example === profileName);
                    });
                    const form = document.getElementById('form');
                    Object.entries(profile).forEach(([name, value]) => {
                        const field = form.elements[name];
                        if (field) field.value = String(value);
                    });
                    form.requestSubmit();
                }

                document.querySelectorAll('.example-button').forEach(button => {
                    button.addEventListener('click', () => fillExample(button.dataset.example));
                });
            </script>
        </body>
        </html>
        """
    )


@app.get('/favicon.ico', include_in_schema=False)
def favicon() -> dict[str, str]:
    return {}


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('telco_churn_api')


class StructuredRequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(json.dumps({
            'event': 'http_request',
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'duration_ms': duration_ms,
        }, ensure_ascii=False))
        response.headers['X-Request-ID'] = request_id
        return response


app.add_middleware(StructuredRequestLogMiddleware)
