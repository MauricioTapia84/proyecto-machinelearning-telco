# Changelog

## Commits

### 15 de octubre de 2026

- #### 3.0. Verificación del punto 01: Problema y metodología

  - Se documentó el problema de negocio como una priorización de clientes con riesgo de churn para apoyar acciones de retención.
  - Se incorporó una tabla que diferencia indicadores de negocio de métricas técnicas.
  - Se calcularon los KPIs iniciales: tasa de churn de 26,54%, Recall del SVM de 77,15% y captura de churn de 51,34% en el 20% de mayor riesgo.
  - Se agregó [scripts/calculate_business_kpis.py](scripts/calculate_business_kpis.py) para reproducir esos indicadores.
  - Se confirmó la existencia del dataset original en `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
  - Se documentó su procedencia como Telco Customer Churn distribuido en Kaggle por `blastchar`, con origen IBM Sample Data Sets.
  - Se verificó que contiene 7.043 registros, 21 columnas y la variable objetivo binaria `Churn` con categorías `Yes` y `No`.
  - Se corrigió en el README la referencia a un archivo procesado inexistente y se reemplazó por las carpetas reales `data/processed/train/` y `data/processed/test/`.
  - Se documentaron las seis etapas de CRISP-DM y la trazabilidad entre cada etapa y los archivos del proyecto.
  - Se aclaró que el despliegue corresponde a la persistencia y uso operativo del pipeline, no a un servicio productivo.
  - Se documentaron los riesgos éticos de sesgo por segmentos, falsos positivos, falsos negativos, privacidad y automatización indebida.
  - Se agregaron medidas de mitigación mediante evaluación segmentada, supervisión humana, explicabilidad y validación controlada de campañas.
  - Se completaron los cuatro subpuntos del punto 01 del checklist.

- #### 3.1. Verificación del punto 02: Análisis exploratorio (EDA)

  - Se confirmó la cobertura individual de las 19 variables predictoras y de la variable objetivo `Churn`, excluyendo correctamente `customerID` como identificador.
  - Se incorporó `SeniorCitizen` al análisis categórico y se verificó que cada predictor tenga visualización, estadísticas y explicación descriptiva.
  - Se corrigieron interpretaciones numéricas desactualizadas para que coincidieran con los cálculos actuales del conjunto de entrenamiento.
  - Se ampliaron las comparaciones frente a `Churn` para todas las variables categóricas mediante tasas de abandono normalizadas por categoría y se agregó un resumen de variables numéricas por estado de abandono.
  - Se verificaron las relaciones entre predictores mediante matriz de correlación y gráficos cruzados de las variables numéricas.
  - Se incorporó un análisis multivariable que combina tipo de contrato, tramo de antigüedad y servicio de internet, con tabla de segmentos y heatmap de tasas de abandono.
  - Se documentaron los principales hallazgos descriptivos y se aclaró que las asociaciones observadas no implican causalidad.
  - Se agregó `tabulate` a [requirements.txt](requirements.txt) y [environment.yml](environment.yml) para garantizar la ejecución reproducible de las tablas Markdown del notebook.
  - Se ejecutó el notebook EDA completo desde cero sin errores, manteniendo el análisis SHAP para la etapa final.
  - Se completaron los seis subpuntos del punto 02 del checklist.

- #### 3.2. Verificación del punto 04: Aprendizaje no supervisado

  - Se verificó la implementación reproducible de PCA y K-Means en el flujo ETL, ajustando las transformaciones únicamente con entrenamiento.
  - Se retuvieron 18 componentes PCA, equivalentes al 95,18% de la varianza, y se generaron tres segmentos mediante K-Means.
  - Se incorporó un perfil interpretable de los segmentos con tamaño, tasa de abandono, antigüedad, cargos, servicios, tipo de contrato, fibra óptica y método de pago.
  - Se documentaron los patrones descriptivos de los tres clusters y se aclaró que el silhouette de 0,0462 refleja una separación débil.
  - Se explicó la utilidad de los segmentos como contexto para priorizar revisiones comerciales, manteniendo el modelo supervisado como mecanismo principal de priorización.
  - Se aclaró que los segmentos no implican causalidad ni permiten estimar por sí solos el efecto futuro de campañas de retención.
  - Se completaron los tres subpuntos del punto 04 del checklist.

- #### 3.3. Verificación del punto 05: Modelos supervisados

  - Se confirmó que el problema corresponde a una clasificación binaria de abandono (`Churn`).
  - Se verificó la implementación de Regresión Logística, Random Forest, Gradient Boosting y Support Vector Machine.
  - Se documentó la justificación de los algoritmos y sus configuraciones de referencia, incluyendo el tratamiento del desbalance mediante SMOTE dentro del entrenamiento.
  - Se corrigió la selección del modelo para utilizar Recall y ROC-AUC promedio de validación cruzada sobre train, manteniendo test reservado para la evaluación final.
  - Se compararon las representaciones originales y PCA + K-Means con métricas adecuadas para el desbalance: Recall, precisión, F1 y ROC-AUC.
  - Se seleccionó Originales + Support Vector Machine, con Recall CV de 0,8222, ROC-AUC CV de 0,8417, Recall test de 0,8091 y ROC-AUC test de 0,8397.
  - Se completaron los cuatro subpuntos del punto 05 del checklist.
