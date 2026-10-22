# Estructura del proyecto

La organización separa datos de origen, matrices procesadas, código reutilizable, notebooks, scripts ejecutables, resultados y documentación.

```text
proyecto/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       └── test/
├── docs/
├── notebooks/
│   ├── 01_EDA_y_Limpieza.ipynb
│   └── 02_Modelado_y_Evaluacion.ipynb
├── output/
│   ├── graficos/
│   ├── modelos/
│   └── resultados/
├── scripts/
│   ├── generate_split.py
│   ├── evaluate_reduced_features.py
│   ├── run_project.py
│   ├── evaluate_operating_scenarios.py
│   ├── evaluate_group_metrics.py
│   └── calculate_business_kpis.py
├── src/
│   ├── processed_data.py
│   ├── modeling.py
│   ├── pipeline.py
│   ├── train.py
│   ├── inference.py
│   ├── decision.py
│   ├── group_metrics.py
│   ├── sensitivity.py
│   ├── split_data.py
│   └── paths.py
├── environment.yml
├── environment-win-64.lock.txt
└── README.md
```

## Responsabilidades

- **`data/raw/`** contiene la fuente original. No se usa directamente para entrenar el modelo canónico.
- **`data/processed/`** contiene los CSV transformados de train y test generados por `generate_split.py`. El modelado consume estas matrices.
- **`src/processed_data.py`** implementa el contrato de partición, deduplicación y generación de matrices transformadas.
- **`src/pipeline.py`** define las transformaciones por distribución: Yeo-Johnson para `TotalCharges`, escalado para numéricas y One-Hot Encoding para categóricas.
- **`src/modeling.py`** implementa el filtro de correlación, SFS, la comparación de modelos y la selección del SVM balanceado.
- **`src/train.py`** ajusta usando solo train y carga test después del ajuste final.
- **`scripts/`** contiene entradas ejecutables; `run_project.py` es la entrada oficial de entrenamiento y persistencia.
- **`notebooks/`** presenta la exploración y la evaluación de forma legible, reutilizando el código de `src/`.
- **`output/`** almacena gráficos, tablas y el artefacto persistido.
- **`docs/`** contiene metodología, coherencia, checklist y material de defensa.

## Flujo reproducible

Desde la raíz del proyecto:

```powershell
conda activate telco-ml
python scripts\generate_split.py
python scripts\evaluate_reduced_features.py
python scripts\run_project.py
```

El primer comando genera los CSV procesados. El segundo documenta correlación, boxplots, SFS y compara la representación completa con la reducida. El tercero selecciona y entrena el modelo final, y solo después evalúa el test reservado.

## Separación metodológica

### Notebook 1: EDA y limpieza

Revisa calidad, distribuciones, sesgo, relaciones con `Churn`, correlaciones y boxplots. Sus variables derivadas son exploratorias y no sustituyen la matriz canónica del modelo.

### Notebook 2: modelado y evaluación

Lee `data/processed/train/`, aplica la reducción de variables sobre train, compara modelos sin SMOTE, ajusta el SVM balanceado y carga `data/processed/test/` únicamente para la evaluación final.

### Inferencia

El artefacto actual espera las columnas transformadas seleccionadas. Para aceptar datos crudos en producción se debe empaquetar explícitamente el preprocesamiento con el clasificador y validar ese nuevo contrato.

## Entorno

Las dependencias directas están en `environment.yml` y la resolución Windows x64 en `environment-win-64.lock.txt`. La ejecución completa está descrita en [README.md](../README.md).
