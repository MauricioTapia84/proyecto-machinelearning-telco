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
| Técnica con uso operativo | Recall de churn | Proporción de los clientes que realmente abandonaron que el modelo identificó como casos positivos. | Verdaderos positivos / (verdaderos positivos + falsos negativos) | **80,91%** en test para el SVM seleccionado | No se inventa una meta; se utiliza como criterio de selección porque el proyecto prioriza no dejar casos de churn sin detectar. |
| Negocio | Captura de churn en el 20% de mayor riesgo | Proporción de todos los abandonos del conjunto de prueba concentrados en el 20% de clientes con mayor probabilidad predicha de churn. | Churn en el 20% superior / total de churn del conjunto de prueba | 194 / 372 = **52,15%** en test | No definida por falta de una meta comercial proporcionada. |

La tasa de churn cuantifica la magnitud inicial del problema. El Recall permite evaluar la capacidad técnica de detectar clientes que efectivamente abandonan. La captura en el 20% superior permitirá responder cuántos abandonos se concentran si el equipo solo puede priorizar una quinta parte de la cartera. Este último indicador no representa una tasa futura de retención: mide únicamente la capacidad de priorización sobre datos históricos de prueba.

El cálculo se reproduce ejecutando el [notebook de modelado](notebooks/02_Modelado_y_Evaluacion.ipynb), que aplica el split estratificado, selecciona el SVM mediante validación cruzada y calcula los indicadores sobre el conjunto de prueba reservado. El script `scripts/calculate_business_kpis.py` queda como apoyo auxiliar para recalcular los KPI cuando el entorno tenga instaladas sus dependencias.

### Etapas CRISP-DM y trazabilidad

El proyecto se organiza según las seis etapas de CRISP-DM. La siguiente tabla muestra qué se realizó y dónde está la evidencia:

| Etapa CRISP-DM | Aplicación en este proyecto | Evidencia |
|---|---|---|
| Comprensión del negocio | Se definió el problema como la priorización de clientes con riesgo de churn para apoyar acciones de retención. Se establecieron objetivos y KPIs. | Esta sección del README y `scripts/calculate_business_kpis.py`. |
| Comprensión de los datos | Se identificó la fuente, la variable objetivo, la estructura del dataset, los tipos, los valores faltantes, el desbalance y los patrones exploratorios. | `notebooks/01_EDA_y_Limpieza.ipynb`. |
| Preparación de los datos | Se eliminaron duplicados, se trataron inconsistencias de `TotalCharges`, se separaron train/test y se aplicaron transformaciones dentro del flujo de modelamiento. | `notebooks/01_EDA_y_Limpieza.ipynb`, `src/pipeline.py` y `data/processed/`. |
| Modelamiento | Se compararon Regresión Logística, Random Forest, Gradient Boosting y SVM, además de las representaciones originales y PCA + K-Means. | `notebooks/02_Modelado_y_Evaluacion.ipynb`. |
| Evaluación | Se utilizaron Recall, Precision, F1-score, ROC-AUC, matrices de confusión, curvas ROC, Precision-Recall, Lift y Ganancia. La selección final priorizó Recall. | `notebooks/02_Modelado_y_Evaluacion.ipynb` y `output/graficos/`. |
| Despliegue/uso | Se persistió el pipeline completo para inferencia posterior y se documentó su uso como apoyo a la priorización comercial. No se presenta como un servicio productivo desplegado. | `output/modelos/`, `src/train.py`, `scripts/train_model.py` y la sección de resultados de este README. |

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
- Duplicados: no hay filas completamente duplicadas incluyendo `customerID`; al excluir ese identificador, se detectan 22 perfiles de variables duplicados, que se eliminan antes del split para evitar repetir observaciones equivalentes.
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
| Eliminación de perfiles duplicados | Antes de separar train/test | Evita que observaciones equivalentes aparezcan en ambos conjuntos y alteren la evaluación. |
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
python scripts\run_project.py
```

Este comando genera las particiones en `data/raw/` y `data/processed/`, entrena el pipeline reutilizable y guarda el modelo en `models/`. Las salidas de los notebooks se conservan en `output/graficos/` y `output/modelos/`.

## 8. Resultados actuales

Se entrenó un pipeline con preprocesamiento, SMOTE y SVM, seleccionado mediante validación cruzada sobre entrenamiento. En el conjunto de prueba se obtuvieron:

- Accuracy: 0.6968
- ROC AUC: aproximadamente 0.84
- Recall para churn: 0.8091
- Precision: 0.4588
- F1-score: 0.5856

El modelo se guarda en:

```text
models/telco_churn_pipeline.joblib
```

Estos resultados apoyan una estrategia de priorización comercial: el Recall permite detectar una proporción alta de los abandonos, mientras que la precisión más baja advierte que se deben gestionar los falsos positivos y definir el umbral con capacidad operativa real.

En una operación, el modelo ordenaría la cartera por probabilidad de churn y apoyaría la decisión de qué clientes priorizar para una revisión preventiva de retención. El equipo podría comenzar por el 20% de mayor riesgo, revisar el contexto del cliente y del segmento, y decidir la acción con supervisión humana. La predicción no determina automáticamente una oferta ni demuestra que una campaña evitará el abandono.

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
