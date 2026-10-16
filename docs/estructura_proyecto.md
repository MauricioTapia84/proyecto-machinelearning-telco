# Estructura actual del proyecto

La estructura versionada separa los datos, el código, los notebooks y los artefactos generados:

project/
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   └── processed/
│       ├── train/
│       └── test/
├── docs/
│   ├── README.md
│   └── estructura_proyecto.md
├── notebooks/
│   ├── 01_EDA_y_Limpieza.ipynb
│   └── 02_Modelado_y_Evaluacion.ipynb
├── output/
│   ├── graficos/
│   └── modelos/
├── scripts/
│   ├── generate_split.py
│   ├── train_model.py
│   └── run_project.py
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── split_data.py
│   └── train.py
├── main.py
├── environment.yml
├── requirements.txt
├── README.md
└── .gitignore

`models/` puede aparecer después de ejecutar `scripts/train_model.py` o `scripts/run_project.py`. Las salidas de los notebooks se almacenan en `output/graficos/` y `output/modelos/`; no se mezclan con los datos originales ni con el código fuente.

## Flujo reproducible

Desde la raíz del proyecto se puede ejecutar `python scripts/run_project.py` para generar las particiones y entrenar el pipeline. Para reproducir el análisis documentado, se ejecutan posteriormente los notebooks en orden.

## Organización metodológica

### Notebook 1
- setup
- carga de datos
- split train/test
- limpieza
- EDA
- transformaciones y escalado

### Notebook 2
- análisis no supervisado
- PCA
- clustering
- modelado supervisado
- evaluación con test
- interpretación de resultados

## Regla clave

La división train/test debe hacerse antes del EDA profundo para evitar data leakage.
