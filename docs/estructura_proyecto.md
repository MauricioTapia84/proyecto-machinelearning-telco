# Estructura Profesional del Proyecto

## Organización recomendada

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
├── models/
├── images/
├── main.py
├── environment.yml
├── requirements.txt
├── README.md
└── .gitignore

## Recomendación metodológica

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
