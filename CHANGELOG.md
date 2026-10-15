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
