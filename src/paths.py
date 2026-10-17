"""Canonical project paths for data and generated artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURES_DIR = OUTPUT_DIR / 'graficos'
MODELS_DIR = OUTPUT_DIR / 'modelos'
RESULTS_DIR = OUTPUT_DIR / 'resultados'
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODEL_PATH = MODELS_DIR / 'telco_churn_pipeline.joblib'
DECISION_SCENARIOS_PATH = RESULTS_DIR / 'escenarios_umbral_24_meses.csv'
GROUP_METRICS_PATH = RESULTS_DIR / 'metricas_por_grupo_test.csv'


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
