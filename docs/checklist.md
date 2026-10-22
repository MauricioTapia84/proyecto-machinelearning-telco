# Checklist de revisión · EP2 Machine Learning

Marcar cada punto cuando exista evidencia en código, notebook o documentación. La revisión integral vigente está en [revision_integral_ep2.md](./revision_integral_ep2.md).

## 01 · Problema y metodología

☑ Problema de negocio, objetivos y KPIs definidos.

☑ Fuente de datos y variable objetivo identificadas.

☑ Etapas de CRISP-DM documentadas.

☑ Impacto ético y límites de uso considerados.

## 02 · Análisis exploratorio

☑ Tipos, nulos, duplicados e inconsistencias revisados.

☑ Variables numéricas y categóricas analizadas con distribuciones, estadísticas y gráficos.

☑ Variables predictoras comparadas con `Churn` mediante tasas y boxplots.

☑ Relaciones entre predictores analizadas con correlación.

☑ `TotalCharges` identificado como variable sesgada y tratado con Yeo-Johnson antes de la estandarización.

☑ Boxplots documentados como evidencia descriptiva, no como prueba causal ni regla automática.

## 03 · Preparación y predictores

☑ Se generan CSV procesados de train/test desde el origen.

☑ La deduplicación se aplica solo a train y el test conserva sus filas.

☑ `TotalCharges` usa Yeo-Johnson; las numéricas adecuadas usan `StandardScaler`; las categóricas usan One-Hot Encoding.

☑ La selección parte de 46 columnas transformadas, filtra correlación absoluta mayor que 0,95 y aplica SFS.

☑ `LinearRegression` se usa solo como estimador de screening con `SequentialFeatureSelector`.

☑ El modelo canónico utiliza 10 variables transformadas seleccionadas.

## 04 · Modelos

☑ Se compara una referencia y varios modelos supervisados.

☑ Recall es la prioridad y ROC-AUC se usa como criterio complementario.

☑ El SVM final usa `class_weight='balanced'` y no SMOTE.

☑ PCA/K-Means queda como comparación exploratoria, no como regla de contacto.

## 05 · Fuga y evaluación

☑ La selección de variables, modelos e hiperparámetros usa solo train.

☑ Test se carga únicamente después del ajuste final.

☑ El test actual tiene 1.408 filas y 363 abandonos.

☑ Se reportan matriz de confusión, Recall, Precision, F1 y ROC-AUC.

☑ Se distingue score de decisión de probabilidad calibrada.

☐ Validación confirmatoria en cohorte temporal o externa no explorada.

## 06 · Reproducibilidad

☑ `generate_split.py`, `evaluate_reduced_features.py` y `run_project.py` forman el flujo documentado.

☑ El artefacto final se persiste en `output/modelos/`.

☑ README, guía, presentación y revisión integral comparten las mismas métricas.

☑ 13 pruebas automatizadas pasan en la configuración verificada.

## 07 · Defensa y uso responsable

☑ Guion de 9 diapositivas organizado como storytelling de ciencia de datos.

☑ Se explican costos de falsos positivos y falsos negativos.

☑ Se declara que priorización histórica no equivale a retención efectiva.

☑ Auditorías por grupo se interpretan de forma descriptiva, no como certificación de equidad.

☐ Ensayo cronometrado y validación externa del material final.

☐ Umbral operativo aprobado con capacidad y costos reales.

☐ Piloto controlado para medir impacto causal.
