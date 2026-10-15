# Proyecto Telco Customer Churn

## Descripción general

Este proyecto desarrolla un modelo de clasificación supervisada para predecir la probabilidad de churn en clientes de telecomunicaciones. El objetivo principal es apoyar decisiones de retención, identificando con anticipación a los clientes con mayor riesgo de abandonar el servicio.

El flujo del proyecto sigue una estructura profesional basada en CRISP-DM, con separación clara entre:

- ETL y limpieza
- partición train/test
- análisis exploratorio
- transformación de variables
- modelado supervisado
- evaluación y validación

## 1. Objetivo del proyecto

Construir y evaluar un modelo que permita detectar clientes con mayor riesgo de abandono a partir de variables demográficas, contractuales, de servicios y de facturación.

### Problema de negocio y KPIs

El problema de negocio es priorizar clientes con mayor riesgo de abandonar el servicio para apoyar acciones preventivas de retención. El modelo no ejecuta campañas ni reemplaza la decisión comercial: entrega una priorización basada en la probabilidad estimada de `Churn`.

Los indicadores se separan entre indicadores de negocio, que describen el tamaño del problema y la utilidad de priorizar clientes, y métricas técnicas, que evalúan el desempeño del modelo.

| Tipo | KPI o métrica | Definición | Fórmula | Valor obtenido | Meta |
|---|---|---|---|---:|---|
| Negocio | Tasa de churn | Proporción de clientes que abandonaron el servicio en el dataset completo. | Clientes con `Churn = Yes` / total de clientes | 1.869 / 7.043 = **26,54%** | No aplica: es la línea base descriptiva del dataset. |
| Técnica con uso operativo | Recall de churn | Proporción de los clientes que realmente abandonaron que el modelo identificó como casos positivos. | Verdaderos positivos / (verdaderos positivos + falsos negativos) | **77,15%** en test para el SVM seleccionado | No se inventa una meta; se utiliza como criterio de selección porque el proyecto prioriza no dejar casos de churn sin detectar. |
| Negocio | Captura de churn en el 20% de mayor riesgo | Proporción de todos los abandonos del conjunto de prueba concentrados en el 20% de clientes con mayor probabilidad predicha de churn. | Churn en el 20% superior / total de churn del conjunto de prueba | 191 / 372 = **51,34%** en test | No definida por falta de una meta comercial proporcionada. |

La tasa de churn cuantifica la magnitud inicial del problema. El Recall permite evaluar la capacidad técnica de detectar clientes que efectivamente abandonan. La captura en el 20% superior permitirá responder cuántos abandonos se concentran si el equipo solo puede priorizar una quinta parte de la cartera. Este último indicador no representa una tasa futura de retención: mide únicamente la capacidad de priorización sobre datos históricos de prueba.

El cálculo se puede reproducir con `python scripts/calculate_business_kpis.py`, utilizando el mismo split estratificado y el pipeline SVM persistido en `output/modelos/support_vector_machine_pipeline.joblib`.

## 2. Fuente de datos

Se utiliza el dataset público Telco Customer Churn de Kaggle.

- Archivo original: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
- Dataset procesado: data/processed/teleco_clean.csv
- Variable objetivo: Churn

La variable objetivo presenta un desbalance relevante, por lo que la evaluación se centra no solo en la precisión global, sino también en métricas orientadas a la detección de clientes en riesgo, especialmente recall y ROC-AUC.

## 3. Stack tecnológico

- Python 3.11
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib
- joblib
- shap

## 4. Estructura del proyecto

```text
project-root/
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
├── images/
├── models/
├── main.py
├── environment.yml
├── requirements.txt
├── README.md
├── .gitignore
└── .git
```

## 5. Flujo de trabajo

### Notebook 1: EDA y limpieza

En este cuaderno se realiza:

- carga de datos
- validación de tipos y estructura
- revisión de valores faltantes
- normalización de tipos
- tratamiento de inconsistencias
- análisis exploratorio
- preparación de datos para modelado

### Notebook 2: modelado y evaluación

En este cuaderno se realiza:

- partición train/test con estratificación
- preprocesamiento
- evaluación de varios modelos
- comparación de métricas
- análisis de clasificación
- interpretación de resultados

## 6. Configuración del entorno con conda

Se recomienda usar conda como entorno principal del proyecto.

```powershell
conda env create -f environment.yml
conda activate telco-ml
```

Si no existe el entorno y se desea crear uno manualmente:

```powershell
conda create -n telco-ml python=3.11 -y
conda activate telco-ml
conda install -c conda-forge pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn shap jupyter ipykernel -y
```

## 7. Ejecución del proyecto

### Generar split train/test

```powershell
python scripts\generate_split.py
```

### Entrenar el modelo

```powershell
python scripts\train_model.py
```

### Ejecutar flujo completo

```powershell
python main.py
```

## 8. Resultados actuales

Se entrenó un pipeline con técnica de preprocesamiento y SVM, obteniendo los siguientes resultados en el conjunto de prueba:

- Accuracy: 0.78
- ROC AUC: 0.8239
- Recall para churn: 0.71

El modelo se guarda en:

```text
models/telco_churn_pipeline.joblib
```

Estos resultados son adecuados para una estrategia de priorización comercial, ya que el objetivo principal es detectar clientes en riesgo con mayor sensibilidad antes de que abandonen el servicio.

## 9. Métricas de negocio y ML

Las métricas evaluadas en el proyecto incluyen:

- Recall
- Precision
- F1-score
- ROC-AUC
- accuracy

Se prioriza el recall porque el costo de un falso negativo es alto: no detectar a un cliente con riesgo real puede provocar pérdida de ingreso y oportunidad de retención.

> “En churn, un falso negativo es más costoso que un falso positivo porque perder un cliente en riesgo representa una pérdida real de ingresos y la oportunidad de intervención.”

Esta prioridad de negocio se refleja en la estrategia final del proyecto: detectar tempranamente a los clientes con riesgo real de abandono permite intervenir con campañas de retención más eficaces, reducir la pérdida de ingresos y mejorar la rentabilidad del negocio.

## 10. Reproducibilidad

El proyecto está estructurado para ser reproducible y reutilizable:

- [environment.yml](environment.yml) define el entorno de trabajo en conda
- [src/pipeline.py](src/pipeline.py) centraliza la lógica reutilizable del preprocesamiento y modelo
- [src/train.py](src/train.py) ejecuta el entrenamiento y guarda el artefacto final
- [scripts/](scripts/) encapsula la ejecución del flujo principal
- [docs/](docs/) almacena la documentación técnica del proyecto

## 11. Consideraciones éticas y de negocio

El modelo debe usarse como herramienta de apoyo para la toma de decisiones, no como reemplazo de la evaluación humana.

Se recomienda:

- revisar la sensibilidad del umbral comercial
- evaluar resultados por segmentos
- controlar falsos positivos en campañas de retención
- mantener la supervisión del equipo comercial
- evitar decisiones automáticas que afecten directamente al cliente sin revisión humana

## 12. Conclusión

El proyecto evidencia una estructura profesional de Machine Learning aplicada a un problema real de churn. La combinación de análisis exploratorio, limpieza, partición estratificada, modelado y evaluación permite construir una solución reproducible y con orientación a negocio.

La decisión final del proyecto se basa en priorizar la detección temprana de clientes en riesgo y actuar con campañas de retención dirigidas, en lugar de centrarse únicamente en una precisión general. Este criterio es clave porque, en churn, un falso negativo es más costoso que un falso positivo: perder a un cliente en riesgo implica una pérdida real de ingresos y la oportunidad de intervención. Por ello, la recomendación de negocio es usar el modelo como apoyo para la toma de decisiones comerciales, priorizando clientes con mayor riesgo y gestionando campañas con un criterio de retención proactiva y sostenible.
