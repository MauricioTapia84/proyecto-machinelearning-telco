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

> **Estado de validación:** el notebook de modelado corregido se ejecutó de principio a fin en un kernel limpio `telco-ml`. La selección programática se hizo con CV en entrenamiento; el test se reportó después y no se utiliza para cambiar modelo o parámetros. Como el EDA describe la cohorte completa, el test no es completamente ciego frente a decisiones humanas informadas por esa exploración; una validación confirmatoria debe usar datos independientes o futuros.

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
| Preparación de los datos | Se normalizaron valores, se aplicó la regla documentada de perfiles repetidos y se reservaron train/test. El modelo canónico vuelve a leer el archivo original y ajusta transformaciones dentro de los folds; no consume las matrices auxiliares exportadas por el EDA. | `notebooks/01_EDA_y_Limpieza.ipynb`, `src/pipeline.py`, `src/modeling.py` y `data/processed/`. |
| Modelamiento | Se compararon Regresión Logística, Random Forest, Gradient Boosting y SVM, además de las representaciones originales y PCA + K-Means. | `notebooks/02_Modelado_y_Evaluacion.ipynb`. |
| Evaluación | El notebook implementa comparación CV con preprocesamiento, PCA/K-Means y SMOTE dentro de folds; Optuna se limita a CV en entrenamiento y el test se reporta al final. Se ejecutó de principio a fin en conda. | `notebooks/02_Modelado_y_Evaluacion.ipynb`; ver [la auditoría](docs/revision_integral_ep2.md). |
| Despliegue/uso | Se plantea como apoyo a la priorización comercial, no como servicio productivo. Notebook y scripts comparten selección, pipeline y artefacto; la ruta de inferencia se valida con pruebas automatizadas. La política comercial sigue pendiente de costos y capacidad reales. | `src/modeling.py`, `src/inference.py`, `scripts/run_project.py` y `output/modelos/`. |

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
- Duplicados: no hay filas completamente duplicadas incluyendo `customerID`; tras normalizar valores vacíos y aplicar la regla de `TotalCharges = 0` cuando `tenure = 0`, se identifican 22 perfiles idénticos entre las 7.043 filas, ignorando el identificador e incluyendo `Churn` en la comparación. Perfiles iguales no prueban que se trate del mismo cliente. La sensibilidad se midió solo en entrenamiento; el test y todas sus copias se excluyeron.

### Sensibilidad a perfiles repetidos

En la partición de entrenamiento, conservar las repeticiones produce 5.636 filas (5.616 perfiles únicos), 20 filas adicionales y 26,53% de churn; deduplicar deja 5.616 filas y 26,44% de churn. Para evitar que las repeticiones inflen la validación, la versión sin deduplicar se evaluó con folds agrupados por perfil.

| Política en train | Recall CV (media ± DE) | ROC-AUC CV (media ± DE) |
|---|---:|---:|
| Con repeticiones, CV agrupada | 0,7940 ± 0,0282 | 0,8474 ± 0,0112 |
| Perfiles deduplicados | 0,7987 ± 0,0287 | 0,8448 ± 0,0111 |

La diferencia es pequeña: deduplicar aumenta el Recall en 0,47 puntos porcentuales y reduce ROC-AUC en 0,26 puntos. Las desviaciones entre folds son mayores que esos cambios; no hay evidencia de una mejora predictiva material por deduplicar. En una CV estratificada ingenua, 17 perfiles repetidos quedarían repartidos entre folds, lo que podría contaminar la evaluación. Se conserva la deduplicación como control contra el sobrepeso de perfiles idénticos, no como afirmación de que sean el mismo cliente. Los valores son sensibilidad de entrenamiento, no resultados del test.
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
- La correlación de Pearson más alta entre predictores numéricos es `tenure`–`TotalCharges` (0,826); `MonthlyCharges`–`TotalCharges` alcanza 0,651. Frente a la codificación binaria de churn, `tenure` se asocia en sentido negativo (−0,352) y `MonthlyCharges` en sentido positivo débil (0,193). Son asociaciones descriptivas, no efectos causales ni probabilidades.
- Las tasas de churn por contrato son 42,7% (mes a mes), 10,7% (un año) y 2,8% (dos años); por servicio de internet son 41,4% (fibra), 19,2% (DSL) y 7,3% (sin internet). El cuaderno no ejecuta una prueba chi-cuadrado; no se afirma significancia estadística.
- El filtro exploratorio de correlación > 0,95 no eliminó columnas. Estas comprobaciones no alteran la lista canónica de 19 predictores.

El detalle de las visualizaciones, tablas y explicaciones se encuentra en [notebooks/01_EDA_y_Limpieza.ipynb](notebooks/01_EDA_y_Limpieza.ipynb).

**Límite de validación:** algunas comparaciones exploratorias utilizan la cohorte completa, incluida `Churn`. El código de selección no consulta el test, pero la exploración humana sí vio esos registros; por eso el resultado de test no es una validación completamente ciega frente a toda decisión analítica. Para una estimación confirmatoria se requiere una cohorte independiente o temporal no explorada.

### Transformaciones aplicadas y justificación

Las transformaciones se aplican de forma reproducible y respetando la separación entre entrenamiento y prueba:

| Transformación | Aplicación | Justificación |
|---|---|---|
| Eliminación de `customerID` | Antes del modelamiento | Es un identificador único y no representa una característica generalizable del cliente. |
| Eliminación de perfiles duplicados | Después de normalizar valores, antes de separar train/test | Aplica la política acordada: perfiles idénticos (incluida la etiqueta) se consideran repetidos aunque cambie `customerID`; se documenta como supuesto, no como identidad confirmada. |
| Conversión de `TotalCharges` | Texto a numérico | Permite realizar estadísticas y transformaciones matemáticas válidas. |
| Tratamiento de `TotalCharges` vacío | `0.0` cuando `tenure = 0`; mediana como respaldo | Los clientes sin antigüedad aún no acumulan cargos; cualquier caso restante se imputa con un estadístico robusto. |
| Variables derivadas | Servicios totales, ratio de cargos por antigüedad, indicador de internet y cliente antiguo | Se generan en el EDA como hipótesis exploratorias; no forman parte del modelo canónico, que conserva los 19 predictores originales. |
| Variables numéricas continuas | Imputación mediana, transformación Yeo-Johnson y escalado cuando corresponde | Reduce el efecto de asimetrías y coloca las variables en una escala comparable. |
| Variables categóricas | Imputación por moda y One-Hot Encoding | Convierte categorías nominales en predictores utilizables sin imponer un orden artificial. |
| Balanceo de clases | SMOTE dentro del entrenamiento | Aumenta la representación de la clase `Churn` sin modificar el conjunto de prueba. |

El ETL del EDA exporta matrices auxiliares para explorar transformaciones y PCA/K-Means, pero el modelo canónico no las consume. El notebook 02 y `scripts/run_project.py` vuelven a leer el CSV original; dentro de la validación cruzada, imputadores, escaladores, codificador, PCA/K-Means cuando corresponde y SMOTE se ajustan solo con el fold de entrenamiento. Esta lógica se comparte mediante [src/pipeline.py](src/pipeline.py) y [src/modeling.py](src/modeling.py).

### Selección inicial de predictores

Antes del modelamiento se conserva como línea base el conjunto completo de **19 predictores originales**: variables demográficas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), antigüedad (`tenure`), servicios contratados, tipo de contrato, facturación, método de pago y cargos (`MonthlyCharges`, `TotalCharges`). Esta decisión se justifica porque todas representan dimensiones disponibles al momento de priorizar clientes y fueron revisadas durante el EDA.

Se excluyen únicamente:

- `customerID`, porque es un identificador único sin capacidad de generalización.
- `Churn`, porque es la variable objetivo y no puede utilizarse como predictor.

Como complemento, el EDA genera cuatro variables derivadas —servicios totales, ratio de cargos por antigüedad, indicador de internet y cliente antiguo— para exploración; no se evalúan como entrada del modelo canónico. La importancia por permutación revisa los 19 predictores originales en un holdout explicativo de entrenamiento, sin modificar la selección del modelo.

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
│   ├── graficos/     # todas las figuras generadas por los notebooks
│   ├── modelos/      # pipeline completo y transformadores exploratorios
│   └── resultados/   # escenarios hipotéticos y métricas por grupo
├── scripts/
│   ├── generate_split.py
│   ├── run_project.py
│   ├── evaluate_operating_scenarios.py
│   └── evaluate_group_metrics.py
├── src/
│   ├── __init__.py
│   ├── decision.py
│   ├── group_metrics.py
│   ├── inference.py
│   ├── modeling.py
│   ├── paths.py
│   ├── pipeline.py
│   ├── sensitivity.py
│   ├── split_data.py
│   └── train.py
├── environment.yml
├── environment-win-64.lock.txt
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
- preparación y exportación de matrices exploratorias; no son entrada del flujo canónico de entrenamiento. Las figuras y transformadores exploratorios se guardan bajo `output/`

### Notebook 2: modelado y evaluación

En este cuaderno se realiza:

- partición train/test con estratificación
- preprocesamiento
- evaluación de varios modelos
- comparación de métricas
- análisis de clasificación
- análisis de sensibilidad a la deduplicación únicamente con datos de entrenamiento
- interpretación de resultados y explicaciones SHAP del score de decisión
- persistencia del pipeline completo en `output/modelos/telco_churn_pipeline.joblib`

## 6. Configuración del entorno con conda

Conda es el gestor y `environment.yml` es la declaración mantenible de las dependencias directas, fijadas a las versiones usadas en el análisis. Para reproducir exactamente la resolución de Windows x64 (incluidas dependencias transitivas y builds), usar el lock explícito versionado:

```powershell
conda create --name telco-ml-repro --file environment-win-64.lock.txt
conda activate telco-ml-repro
python -m ipykernel install --user --name telco-ml-repro --display-name "Python (telco-ml reproducible)"
```

Si se desea resolver el entorno para otro sistema operativo desde la especificación directa:

```powershell
conda env create -f environment.yml
```

Versiones principales validadas: Python **3.11.17**, pandas **3.0.6**, NumPy **2.4.6**, scikit-learn **1.9.1**, imbalanced-learn **0.14.2**, SHAP **0.51.0**, Optuna **5.0.0**, Jupyter **1.1.1** e ipykernel **7.4.0**. `requirements.txt` se retiró para evitar mantener dos manifiestos de paquetes diferentes.

Al modificar el entorno Windows, regenerar el lock desde el entorno validado con `conda list -n telco-ml --explicit | Set-Content -Encoding ascii environment-win-64.lock.txt`. En VS Code, seleccionar el kernel conda recién creado. Si PowerShell indica que `conda` no se reconoce, abrir Miniforge Prompt.

## 7. Ejecución del proyecto

### Flujo canónico de selección, entrenamiento y persistencia

```powershell
conda activate telco-ml
python scripts\run_project.py
```

`scripts/run_project.py` es la única entrada oficial para entrenar, evaluar y guardar el modelo. La lógica reutilizable vive en `src/`; no se mantienen comandos alternativos de entrenamiento en la raíz ni en `scripts/`.

Este comando aplica la misma preparación, split estratificado, selección por validación cruzada, búsqueda Optuna y guardado del pipeline que el notebook 02. El test se consulta solo al final para el reporte; no se utiliza para elegir el modelo. El artefacto incluye imputación, codificación, escalado, balanceo y clasificador.

Para cargarlo y puntuar nuevos clientes, usar `src.inference.load_model_artifact` y `src.inference.score_customers`. Los nuevos registros deben contener los predictores presentes al entrenar; `customerID` y `Churn` (si vienen) no se usan como predictores.

`scripts\generate_split.py` conserva una función auxiliar para exportar particiones procesadas del EDA; no es un paso previo del flujo canónico.

## 8. Resultados actuales

La evaluación corregida seleccionó el **SVM lineal ajustado por Optuna** con validación cruzada estratificada de cinco folds, priorizando Recall y usando ROC-AUC como desempate. El candidato obtuvo Recall CV **0,8007 ± 0,0388** y ROC-AUC CV **0,8471 ± 0,0186**. Optuna evaluó cinco combinaciones de hiperparámetros usando únicamente entrenamiento; la configuración elegida fue `C=0,7821`, `class_weight=balanced`.

En la evaluación final, realizada después de cerrar la selección, se obtuvieron Accuracy **0,7445**, Recall de churn **0,7715**, Precision de churn **0,5116**, F1 de churn **0,6152** y ROC-AUC **0,8405**. La matriz de confusión tuvo 759 verdaderos negativos, 274 falsos positivos, 85 falsos negativos y 287 verdaderos positivos. En el 20% superior del ranking de riesgo quedaron 192 de los 372 casos de churn (51,61%).

El ranking usa el score de decisión del SVM, que **no es una probabilidad calibrada**. El notebook y el comando canónico usan el mismo selector, pipeline y contrato de artefacto. Las nuevas métricas deben actualizarse después de ejecutar ambos con la misma versión de dependencias.

Estos resultados son descriptivos del test reservado. La precisión de 0,5116 indica que todavía hay falsos positivos; el umbral del clasificador no constituye una política de campaña y debe contrastarse con capacidad y costos reales.

En una operación, el modelo ordenaría la cartera por score de riesgo de churn y apoyaría la decisión de qué clientes priorizar para una revisión preventiva de retención. El 20% superior se presenta solo como escenario descriptivo; no se ha aprobado como capacidad de campaña. El equipo debe revisar el contexto del cliente y decidir la acción con supervisión humana. La predicción no determina automáticamente una oferta ni demuestra que una campaña evitará el abandono.

Se construyó una **política operativa hipotética** con scores fuera de fold de entrenamiento: elegir el umbral que minimiza el costo proxy bajo un máximo de contactos. El umbral calculado es solo para análisis de escenarios; el artefacto mantiene `operating_threshold=None` y el score SVM no calibrado no debe leerse como probabilidad individual.

### Escenarios operativos hipotéticos

El script `scripts/evaluate_operating_scenarios.py` usa capacidad de 250 contactos por cada 7.043 clientes, eficacia hipotética de 20%, costo de contacto de 1%, 5% o 10% de la mediana de `MonthlyCharges`, horizonte de 24 meses y valor incremental retenido igual a cero. La pérdida evitada se aproxima con 24 veces la mediana de cargos mensuales (**1.689,6 unidades monetarias por churn**); es facturación bruta, no margen ni utilidad.

| Costo de contacto | Umbral OOF | Contactos OOF / cupo 199 | Contactos test / cupo 50 | Pérdida bruta evitada test* | Costo contactos test | Pérdida neta proxy test | Ahorro proxy vs. no contactar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% de mediana mensual | 1,7907 | 198 | 44 | 13.516,80 | 30,98 | 615.045,38 | 13.485,82 |
| 5% de mediana mensual | 1,7907 | 198 | 44 | 13.516,80 | 154,88 | 615.169,28 | 13.361,92 |
| 10% de mediana mensual | 1,7907 | 198 | 44 | 13.516,80 | 309,76 | 615.324,16 | 13.207,04 |

En la lista de test, 40 de 44 contactos abandonaron históricamente; con 20% de eficacia supuesta, el escenario estima 8 abandonos evitados, no observados. La pérdida proxy sin campaña para los 372 abandonos de test sería 628.531,20 unidades; después de la intervención hipotética y el costo de contacto, la pérdida neta se estima entre 615.045,38 y 615.324,16. El umbral se eligió con OOF train, no con test; los valores test son un reporte ilustrativo posterior.

**Interpretación:** el escenario genera ahorro proxy positivo porque se impuso una eficacia de 20% y se usa una proxy de ingreso bruto. No es evidencia de ahorro real ni de causalidad. La moneda no está especificada, no se dispone de margen, costo de ofertas, costo de servicio, tasa de respuesta ni valor de vida; la capacidad también es supuesta. No fijar ni desplegar este umbral como política real sin reemplazar esos supuestos y validar una campaña controlada. El detalle reproducible está en [output/resultados/escenarios_umbral_24_meses.csv](output/resultados/escenarios_umbral_24_meses.csv).

### Errores del modelo por grupos

El script `scripts/evaluate_group_metrics.py` aplica el clasificador ya fijado al mismo test reservado y reporta conteos, Recall, precisión, tasa de falsos positivos, tasa de selección e intervalos Wilson del 95% para género, `SeniorCitizen`, contrato, servicio de internet y tramos de antigüedad. Usa el `predict` por defecto del SVM, **no** el umbral hipotético anterior.

| Grupo | Recall | Precisión | Tasa de falsos positivos | Soporte churn |
|---|---:|---:|---:|---:|
| Female | 0,763 | 0,535 | 0,262 | 190 |
| Male | 0,780 | 0,490 | 0,268 | 182 |
| `SeniorCitizen=0` | 0,728 | 0,501 | 0,230 | 283 |
| `SeniorCitizen=1` | 0,910 | 0,540 | 0,493 | 89 |
| Contrato mes a mes | 0,864 | 0,516 | 0,600 | 330 |
| Contrato de un año | 0,059 | 0,250 | 0,023 | 34 |
| Contrato de dos años | 0,000 | 0,000 | 0,003 | 8 |

Las diferencias —en especial la tasa de falsos positivos de `SeniorCitizen=1`, y el Recall bajo en contratos largos— justifican seguimiento y revisión antes de cualquier uso operativo. El grupo de contrato de dos años tiene solo 8 abandonos; sus estimaciones son muy inciertas. Los intervalos y los conteos por todos los grupos están en [output/resultados/metricas_por_grupo_test.csv](output/resultados/metricas_por_grupo_test.csv). Esta evaluación es descriptiva y post hoc: no prueba discriminación, no define por sí sola un umbral justo y no debe usarse para volver a ajustar el modelo contra este test.

### Decisión sobre K-Means

K-Means queda **solo como análisis exploratorio y comparador metodológico**, no como mecanismo de segmentación ni de contacto comercial. La silhouette de **0,0462** refleja separación débil; además, el SVM con predictores originales obtiene Recall CV **0,8007**, frente a **0,7751** para la representación PCA + K-Means. No hay evidencia suficiente de segmentos estables, accionables o superiores al flujo supervisado.

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

- [environment.yml](environment.yml) define las dependencias directas fijadas; [environment-win-64.lock.txt](environment-win-64.lock.txt) registra la resolución exacta para Windows x64
- [src/pipeline.py](src/pipeline.py) centraliza la lógica reutilizable del preprocesamiento
- [src/modeling.py](src/modeling.py) comparte la selección y persistencia entre notebook y scripts
- [src/inference.py](src/inference.py) carga el artefacto y puntúa registros nuevos
- [src/decision.py](src/decision.py) evalúa umbrales bajo costos y capacidad usando scores fuera de fold
- [src/group_metrics.py](src/group_metrics.py) calcula métricas e intervalos descriptivos por grupo
- [scripts/evaluate_operating_scenarios.py](scripts/evaluate_operating_scenarios.py) cuantifica pérdidas/ahorros proxy bajo supuestos hipotéticos
- [scripts/evaluate_group_metrics.py](scripts/evaluate_group_metrics.py) reporta errores segmentados en el test reservado
- [src/paths.py](src/paths.py) define las rutas canónicas bajo `output/`
- [scripts/run_project.py](scripts/run_project.py) ejecuta la selección, el entrenamiento y la evaluación final
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

La evaluación demuestra capacidad predictiva y de priorización en este dataset, pero no establece que un falso negativo sea más costoso que un falso positivo ni que una campaña retenga clientes. Antes de elegir un umbral o una lista operativa, el área de negocio debe aportar la capacidad de contacto por periodo, el costo por contacto, la pérdida esperada por churn y la eficacia/valor de una intervención exitosa. Hasta contar con esos datos, el score solo debe usarse para ordenar casos en análisis retrospectivo, sin convertir el 20% superior ni el umbral por defecto en una política aprobada.
