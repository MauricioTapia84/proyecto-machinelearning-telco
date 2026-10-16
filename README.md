# Proyecto Telco Customer Churn

## Descripción general

Este proyecto desarrolla modelos de clasificación supervisada para ordenar clientes de telecomunicaciones por score de riesgo de churn y apoyar decisiones de retención. El score del SVM seleccionado no está calibrado como probabilidad.

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

El problema de negocio es priorizar clientes con mayor riesgo de abandonar el servicio para apoyar acciones preventivas de retención. El modelo no ejecuta campañas ni reemplaza la decisión comercial: entrega un score de riesgo para ordenar clientes; no equivale necesariamente a una probabilidad calibrada de `Churn`.

Los indicadores se separan entre indicadores de negocio, que describen el tamaño del problema y la utilidad de priorizar clientes, y métricas técnicas, que evalúan el desempeño del modelo.

> **Estado de validación:** el notebook de modelado corregido se ejecutó de principio a fin en un kernel limpio `telco-ml`. Los valores de evaluación del modelo en esta sección corresponden a esa ejecución. La selección se hizo con CV en entrenamiento; el test se reportó después y no debe reutilizarse para modificar el modelo o sus parámetros.

| Tipo | KPI o métrica | Definición | Fórmula | Valor obtenido | Meta |
|---|---|---|---|---:|---|
| Negocio | Tasa de churn | Proporción de clientes que abandonaron el servicio en el dataset completo. | Clientes con `Churn = Yes` / total de clientes | 1.869 / 7.043 = **26,54%** | No aplica: es la línea base descriptiva del dataset. |
| Técnica con uso operativo | Recall de churn | Proporción de los clientes que realmente abandonaron que el modelo identificó como casos positivos al umbral por defecto del clasificador. | Verdaderos positivos / (verdaderos positivos + falsos negativos) | **77,15%** (287 / 372) en test; SVM lineal ajustado | Se priorizó Recall en CV, pero el umbral no representa todavía una regla comercial. |
| Técnica | ROC-AUC de churn | Capacidad de ordenar churn sobre no churn a través de distintos umbrales. | Área bajo la curva ROC | **0,8405** en test; CV de 5 folds: 0,8471 ± 0,0186 para el candidato seleccionado | No definida. |
| Negocio | Captura de churn en el 20% de mayor riesgo | Proporción de todos los abandonos del conjunto de prueba entre los 281 clientes con mayor score de riesgo. | Churn en el 20% superior / total de churn del conjunto de prueba | 192 / 372 = **51,61%** en test; Lift **2,58** frente a selección aleatoria | No definida por falta de capacidad/costo comercial acordados. |

La tasa de churn cuantifica la magnitud inicial del problema. El Recall permite evaluar la capacidad técnica de detectar clientes que efectivamente abandonan. La captura en el 20% superior permitirá responder cuántos abandonos se concentran si el equipo solo puede priorizar una quinta parte de la cartera. Este último indicador no representa una tasa futura de retención: mide únicamente la capacidad de priorización sobre datos históricos de prueba.

### Etapas CRISP-DM y trazabilidad

El proyecto se organiza según las seis etapas de CRISP-DM. La siguiente tabla muestra qué se realizó y dónde está la evidencia:

| Etapa CRISP-DM | Aplicación en este proyecto | Evidencia |
|---|---|---|
| Comprensión del negocio | Se definió el problema como la priorización de clientes con riesgo de churn para apoyar acciones de retención. Se establecieron objetivos y KPIs. | Esta sección del README y `scripts/calculate_business_kpis.py`. |
| Comprensión de los datos | Se identificó la fuente, la variable objetivo, la estructura del dataset, los tipos, los valores faltantes, el desbalance y los patrones exploratorios. | `notebooks/01_EDA_y_Limpieza.ipynb`. |
| Preparación de los datos | Se eliminaron duplicados, se trataron inconsistencias de `TotalCharges`, se separaron train/test y se aplicaron transformaciones dentro del flujo de modelamiento. | `notebooks/01_EDA_y_Limpieza.ipynb`, `src/pipeline.py` y `data/processed/`. |
| Modelamiento | Se compararon Regresión Logística, Random Forest, Gradient Boosting y SVM, además de las representaciones originales y PCA + K-Means. | `notebooks/02_Modelado_y_Evaluacion.ipynb`. |
| Evaluación | El notebook implementa comparación CV con preprocesamiento, PCA/K-Means y SMOTE dentro de folds; Optuna se limita a CV en entrenamiento y el test se reporta al final. Se ejecutó de principio a fin en conda. | `notebooks/02_Modelado_y_Evaluacion.ipynb`; ver [la auditoría](docs/revision_integral_ep2.md). |
| Despliegue/uso | Se plantea como apoyo a la priorización comercial, no como servicio productivo. El artefacto de inferencia completo y un único flujo reproducible siguen pendientes de implementación/verificación. | `src/`, `scripts/` y [la auditoría](docs/revision_integral_ep2.md). |

Esta trazabilidad permite distinguir entre lo que fue ejecutado en el proyecto y lo que queda como una posible evolución futura, especialmente el despliegue productivo y la medición del impacto real de campañas de retención.

## 2. Fuente de datos

Se utiliza el dataset público **Telco Customer Churn** distribuido en Kaggle por `blastchar`, cuyo origen y publicación educativa se atribuyen a IBM Sample Data Sets.

- Archivo original: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
- Datos procesados: data/processed/train/ y data/processed/test/
- Variable objetivo: Churn

El archivo original contiene 7.043 clientes y 21 columnas: 20 variables predictoras y la variable objetivo `Churn`. Esta última es binaria y utiliza `Yes` para abandono y `No` para permanencia. En el dataset se observan 1.869 abandonos y 5.174 permanencias.

### Control de calidad del dataset

La revisión independiente del archivo original confirmó:

- Tipos: `SeniorCitizen` y `tenure` son enteros; `MonthlyCharges` es decimal; `TotalCharges` llega como texto por contener espacios en blanco.
- Nulos: no hay valores `NaN` explícitos, pero existen 11 espacios en blanco ocultos en `TotalCharges`; todos corresponden a clientes con `tenure = 0`.
- Duplicados: no hay filas completamente duplicadas incluyendo `customerID`; tras normalizar valores vacíos y aplicar la regla de `TotalCharges = 0` cuando `tenure = 0`, el proyecto elimina 22 perfiles idénticos antes del split, ignorando el identificador e incluyendo `Churn` en la comparación. Es una decisión analítica: perfiles iguales no prueban que se trate del mismo cliente. Se debe contrastar el resultado con una ejecución que conserve perfiles con distinto `customerID`.
- Inconsistencias: las categorías son válidas según el diccionario del dataset, `SeniorCitizen` solo contiene 0/1 y no se detectaron valores fuera de rango en `tenure` o `MonthlyCharges`.

La limpieza convierte `TotalCharges` a numérico, asigna 0 a los casos con `tenure = 0` y utiliza la mediana solo como respaldo para cualquier nulo remanente.

La variable objetivo presenta un desbalance relevante, por lo que la evaluación se centra no solo en la precisión global, sino también en métricas orientadas a la detección de clientes en riesgo, especialmente recall y ROC-AUC.

### Resumen del análisis exploratorio

El EDA identificó los siguientes patrones descriptivos relevantes:

- La tasa histórica de churn es 26,54% y la clase `Churn = Yes` está desbalanceada frente a la permanencia.
- Los contratos mes a mes presentan mayor abandono que los contratos de uno o dos años.
- Los clientes con menor antigüedad concentran una proporción mayor de abandono que los clientes antiguos.
- Se observaron diferencias de churn asociadas a servicios de internet, método de pago y cargos mensuales; estas relaciones se utilizan para orientar la exploración y no se interpretan como causalidad.
- Las variables numéricas fueron revisadas con distribuciones, boxplots, estadísticas descriptivas y comparaciones frente a `Churn`; las variables categóricas se analizaron con frecuencias, porcentajes y tasas de abandono por categoría.

El detalle de las visualizaciones, tablas y explicaciones se encuentra en [notebooks/01_EDA_y_Limpieza.ipynb](notebooks/01_EDA_y_Limpieza.ipynb).

### Transformaciones aplicadas y justificación

Las transformaciones se aplican de forma reproducible y respetando la separación entre entrenamiento y prueba:

| Transformación | Aplicación | Justificación |
|---|---|---|
| Eliminación de `customerID` | Antes del modelamiento | Es un identificador único y no representa una característica generalizable del cliente. |
| Eliminación de perfiles duplicados | Después de normalizar valores, antes de separar train/test | Aplica la política acordada: perfiles idénticos (incluida la etiqueta) se consideran repetidos aunque cambie `customerID`; se documenta como supuesto, no como identidad confirmada. |
| Conversión de `TotalCharges` | Texto a numérico | Permite realizar estadísticas y transformaciones matemáticas válidas. |
| Tratamiento de `TotalCharges` vacío | `0.0` cuando `tenure = 0`; mediana como respaldo | Los clientes sin antigüedad aún no acumulan cargos; cualquier caso restante se imputa con un estadístico robusto. |
| Variables derivadas | Servicios totales, ratio de cargos por antigüedad, indicador de internet y cliente antiguo | Resume información operativa y permite representar relaciones potencialmente útiles para el modelo. |
| Variables numéricas continuas | Imputación mediana, transformación Yeo-Johnson y escalado cuando corresponde | Reduce el efecto de asimetrías y coloca las variables en una escala comparable. |
| Variables categóricas | Imputación por moda y One-Hot Encoding | Convierte categorías nominales en predictores utilizables sin imponer un orden artificial. |
| Balanceo de clases | SMOTE dentro del entrenamiento | Aumenta la representación de la clase `Churn` sin modificar el conjunto de prueba. |

La mediana, los escaladores, el codificador y las demás transformaciones estadísticas se ajustan únicamente con entrenamiento y luego se aplican a prueba. Esta lógica está implementada en [src/pipeline.py](src/pipeline.py) y en el ETL de [notebooks/01_EDA_y_Limpieza.ipynb](notebooks/01_EDA_y_Limpieza.ipynb).

### Selección inicial de predictores

Antes del modelamiento se conserva como línea base el conjunto completo de **19 predictores originales**: variables demográficas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), antigüedad (`tenure`), servicios contratados, tipo de contrato, facturación, método de pago y cargos (`MonthlyCharges`, `TotalCharges`). Esta decisión se justifica porque todas representan dimensiones disponibles al momento de priorizar clientes y fueron revisadas durante el EDA.

Se excluyen únicamente:

- `customerID`, porque es un identificador único sin capacidad de generalización.
- `Churn`, porque es la variable objetivo y no puede utilizarse como predictor.

Como complemento se generan cuatro variables derivadas —servicios totales, ratio de cargos por antigüedad, indicador de internet y cliente antiguo— porque resumen relaciones de negocio previamente observadas. No se eliminan predictores originales en esta etapa: la relevancia posterior y la comparación contra esta línea base se revisarán después del modelamiento.

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

Se recomienda usar conda como entorno principal del proyecto. Desde la raíz del repositorio:

```powershell
conda env create -f environment.yml
conda activate telco-ml
python -m ipykernel install --user --name telco-ml --display-name "Python (telco-ml conda)"
```

En VS Code, seleccionar el kernel **Python (telco-ml conda)** en ambos notebooks. `environment.yml` es la definición del entorno; después de cambiar dependencias, recrearlo o actualizarlo desde ese archivo.
Si PowerShell indica que `conda` no se reconoce, abrir **Miniforge Prompt** y ejecutar allí los comandos; en VS Code hay que seleccionar el kernel del entorno, no el Python del sistema.

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
python scripts\run_project.py
```

**Estado:** estos scripts todavía no constituyen un flujo canónico verificado: la generación del split y el entrenamiento no consumen necesariamente la misma partición, y hay diferencias de rutas y artefactos. No se debe interpretar esta secuencia como reproducción de las métricas anteriores hasta completar la unificación descrita en la [auditoría](docs/revision_integral_ep2.md).

## 8. Resultados actuales

La evaluación corregida seleccionó el **SVM lineal ajustado por Optuna** con validación cruzada estratificada de cinco folds, priorizando Recall y usando ROC-AUC como desempate. El candidato obtuvo Recall CV **0,8007 ± 0,0388** y ROC-AUC CV **0,8471 ± 0,0186**. Optuna evaluó cinco combinaciones de hiperparámetros usando únicamente entrenamiento; la configuración elegida fue `C=0,7821`, `class_weight=balanced`.

En la evaluación final, realizada después de cerrar la selección, se obtuvieron Accuracy **0,7445**, Recall de churn **0,7715**, Precision de churn **0,5116**, F1 de churn **0,6152** y ROC-AUC **0,8405**. La matriz de confusión tuvo 759 verdaderos negativos, 274 falsos positivos, 85 falsos negativos y 287 verdaderos positivos. En el 20% superior del ranking de riesgo quedaron 192 de los 372 casos de churn (51,61%).

El ranking usa el score de decisión del SVM, que **no es una probabilidad calibrada**. Estos resultados se encuentran en el notebook actualizado. El script de entrenamiento de `src/` es todavía un flujo aparte; el modelo persistido allí no debe asumirse idéntico al candidato de esta evaluación. Aún falta unificar el entrenamiento oficial y validar la persistencia/inferencia del artefacto seleccionado.

Estos resultados son descriptivos del test reservado. La precisión de 0,5116 indica que todavía hay falsos positivos; el umbral del clasificador no constituye una política de campaña y debe contrastarse con capacidad y costos reales.

En una operación, el modelo ordenaría la cartera por score de riesgo de churn y apoyaría la decisión de qué clientes priorizar para una revisión preventiva de retención. El equipo podría comenzar por el 20% de mayor riesgo, revisar el contexto del cliente y del segmento, y decidir la acción con supervisión humana. La predicción no determina automáticamente una oferta ni demuestra que una campaña evitará el abandono.

## 9. Métricas de negocio y ML

Las métricas evaluadas en el proyecto incluyen:

- Recall
- Precision
- F1-score
- ROC-AUC
- accuracy

El análisis priorizó Recall bajo la hipótesis de que perder un cliente en riesgo puede ser más costoso que contactar a un cliente que finalmente permanece. Esa hipótesis aún no se ha cuantificado con costos reales ni capacidad de campaña y debe validarse con el área de negocio.

> Hipótesis por validar: en la operación de churn, un falso negativo podría tener un costo mayor que un falso positivo. La magnitud relativa de esos costos no se ha medido en este proyecto.

Las predicciones podrían ayudar a priorizar clientes para una intervención, pero las métricas predictivas no demuestran que las campañas eviten abandonos ni que aumenten la rentabilidad. Ese impacto tendría que medirse mediante una prueba controlada o seguimiento operativo.

## 10. Reproducibilidad

El proyecto está estructurado para ser reproducible y reutilizable:

- [environment.yml](environment.yml) define el entorno de trabajo en conda
- [src/pipeline.py](src/pipeline.py) centraliza la lógica reutilizable del preprocesamiento y modelo
- [src/train.py](src/train.py) ejecuta el entrenamiento y guarda el artefacto final
- [scripts/](scripts/) encapsula la ejecución del flujo principal
- [docs/](docs/) almacena la documentación técnica del proyecto

## 11. Consideraciones éticas y de negocio

El dataset utilizado es sintético y educativo, por lo que no contiene registros reales de clientes. Si el modelo se trasladara a datos reales, se deberían aplicar controles de privacidad, acceso restringido, minimización de datos y cumplimiento de la normativa vigente.

Los principales riesgos identificados son:

- **Sesgo por segmentos:** variables como `gender`, `SeniorCitizen`, `tenure` y `MonthlyCharges` pueden generar diferencias de error o de priorización entre grupos.
- **Falsos positivos:** un cliente clasificado como riesgoso podría recibir ofertas o contactos innecesarios, generando molestias o un uso ineficiente de recursos.
- **Falsos negativos:** un cliente con riesgo real podría no ser contactado, perdiéndose una oportunidad de retención.
- **Automatización indebida:** una predicción de churn no debe utilizarse para negar servicios, aumentar precios o tomar decisiones adversas sin revisión humana.

Las medidas de mitigación consideradas son:

- revisar la sensibilidad del umbral comercial
- evaluar Recall, precisión y tasas de error por segmentos
- controlar falsos positivos en campañas de retención
- mantener la supervisión del equipo comercial
- evitar decisiones automáticas que afecten directamente al cliente sin revisión humana
- utilizar explicaciones SHAP como apoyo para revisar predicciones y detectar comportamientos inesperados
- validar cualquier campaña real mediante seguimiento o un grupo de comparación, porque el modelo predictivo no demuestra por sí solo que una intervención evite el abandono

## 12. Conclusión

El proyecto evidencia una estructura profesional de Machine Learning aplicada a un problema real de churn. La combinación de análisis exploratorio, limpieza, partición estratificada, modelado y evaluación permite construir una solución reproducible y con orientación a negocio.

La decisión final del proyecto se basa en priorizar la detección temprana de clientes en riesgo y actuar con campañas de retención dirigidas, en lugar de centrarse únicamente en una precisión general. Este criterio es clave porque, en churn, un falso negativo es más costoso que un falso positivo: perder a un cliente en riesgo implica una pérdida real de ingresos y la oportunidad de intervención. Por ello, la recomendación de negocio es usar el modelo como apoyo para la toma de decisiones comerciales, priorizando clientes con mayor riesgo y gestionando campañas con un criterio de retención proactiva y sostenible.
