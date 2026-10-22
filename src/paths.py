"""Canonical project paths for data and generated artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURES_DIR = OUTPUT_DIR / 'graficos'
MODELS_DIR = OUTPUT_DIR / 'modelos'
RESULTS_DIR = OUTPUT_DIR / 'resultados'
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_TRAIN_DIR = PROCESSED_DATA_DIR / 'train'
PROCESSED_TEST_DIR = PROCESSED_DATA_DIR / 'test'
PROCESSED_TRAIN_X_ORIGINAL_PATH = PROCESSED_TRAIN_DIR / 'X_original.csv'
PROCESSED_TEST_X_ORIGINAL_PATH = PROCESSED_TEST_DIR / 'X_original.csv'
PROCESSED_TRAIN_X_PCA_PATH = PROCESSED_TRAIN_DIR / 'X_pca_kmeans.csv'
PROCESSED_TEST_X_PCA_PATH = PROCESSED_TEST_DIR / 'X_pca_kmeans.csv'
PROCESSED_TRAIN_Y_PATH = PROCESSED_TRAIN_DIR / 'y.csv'
PROCESSED_TEST_Y_PATH = PROCESSED_TEST_DIR / 'y.csv'
MODEL_PATH = MODELS_DIR / 'telco_churn_pipeline.joblib'
DECISION_SCENARIOS_PATH = RESULTS_DIR / 'escenarios_umbral_24_meses.csv'
GROUP_METRICS_PATH = RESULTS_DIR / 'metricas_por_grupo_test.csv'
FEATURE_SELECTION_PATH = RESULTS_DIR / 'seleccion_variables_sfs.csv'
CORRELATION_MATRIX_PATH = RESULTS_DIR / 'matriz_correlacion_train.csv'
FEATURE_BOXPLOT_PATH = FIGURES_DIR / 'boxplots_variables_seleccionadas_vs_churn.png'


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
