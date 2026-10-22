# Revisión integral de coherencia · EP2

## Dictamen

El proyecto tiene un flujo reproducible y coherente después de consolidar la selección de variables. La fuente del modelado es `data/processed/`; el conjunto de test permanece reservado hasta la evaluación final. La configuración vigente es:

```text
46 columnas transformadas
  → filtro de correlación absoluta > 0,95
  → SequentialFeatureSelector forward, 5 folds, 10 columnas
  → SVM lineal balanceado ajustado por Optuna
  → sin SMOTE
```

La documentación de defensa y presentación fue actualizada para reflejar esta configuración. Las afirmaciones históricas sobre SMOTE como parte del flujo principal, el split de 7.021 filas, el Recall de test 0,7715 y el ROC-AUC 0,8405 ya no deben utilizarse como resultados vigentes.

## Matriz de coherencia entre etapas

| Etapa | Decisión vigente | Evidencia | Estado |
|---|---|---|---|
| Negocio | Priorizar revisiones humanas de retención; no automatizar campañas | [README.md](../README.md), [presentacion_ep2.md](./presentacion_ep2.md) | Coherente |
| Datos | 7.043 filas, 19 predictores originales, 1.869 churn | [01_EDA_y_Limpieza.ipynb](../notebooks/01_EDA_y_Limpieza.ipynb) | Coherente |
| Calidad | `TotalCharges` textual, 11 espacios, perfiles repetidos controlados en train | [processed_data.py](../src/processed_data.py) | Coherente |
| Transformación | Yeo-Johnson para `TotalCharges`, StandardScaler para numéricas adecuadas, One-Hot para categorías | [pipeline.py](../src/pipeline.py), [guia_defensa_ep2.md](./guia_defensa_ep2.md) | Coherente |
| Partición | 5.617 train y 1.408 test; holdout por perfil | [generate_split.py](../scripts/generate_split.py), [split_data.py](../src/split_data.py) | Coherente |
| Selección | Correlación > 0,95 y SFS con `LinearRegression` como screening | [modeling.py](../src/modeling.py), [seleccion_variables_sfs.csv](../output/resultados/seleccion_variables_sfs.csv) | Coherente |
| Balanceo | `class_weight='balanced'`; no SMOTE | [modeling.py](../src/modeling.py), [run_project.py](../scripts/run_project.py) | Coherente |
| Modelo final | SVM lineal ajustado por Optuna, prioridad Recall | [train.py](../src/train.py), [telco_churn_pipeline.joblib](../output/modelos/telco_churn_pipeline.joblib) | Coherente |
| Test | Se carga solo después del ajuste final | [train.py](../src/train.py), [evaluate_reduced_features.py](../scripts/evaluate_reduced_features.py) | Coherente |
| Resultado | El modelo reducido conserva las métricas de clasificación y pierde 0,0028 ROC-AUC frente al completo | [comparacion_variables_reducidas.csv](../output/resultados/comparacion_variables_reducidas.csv) | Coherente |

## Resultado vigente

| Modelo | Variables | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM completo | 46 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8474 |
| SVM reducido | 10 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8445 |

La matriz del modelo reducido es `[[699, 346], [57, 306]]`. El test tiene 363 casos churn. Estas métricas son descriptivas del split actual y no prueban impacto causal de una campaña.

## Reglas para no romper la coherencia

1. Regenerar primero los CSV transformados con `python scripts\generate_split.py`.
2. Ejecutar selección/evaluación con `python scripts\evaluate_reduced_features.py`.
3. Entrenar y persistir con `python scripts\run_project.py`.
4. No abrir `data/processed/test/` para seleccionar variables, modelos o hiperparámetros.
5. Si se modifica la semilla, el split, el estimador de SFS o el tratamiento de clases, regenerar resultados y actualizar README, guía y presentación juntos.
6. Mantener la distinción entre score de decisión, probabilidad calibrada, priorización histórica y retención efectiva.

## Validaciones realizadas

- 13 pruebas automatizadas pasan en la configuración local.
- Los módulos de `src/`, `scripts/` y `tests/` compilan correctamente.
- Los notebooks mantienen celdas ejecutables válidas.
- Los resultados reducidos se guardan en [output/resultados/](../output/resultados/) y el artefacto en [output/modelos/](../output/modelos/).

## Pendientes explícitos

- Confirmar resultados en una cohorte temporal o externa no explorada.
- Evaluar calibración si el negocio necesita probabilidades.
- Definir capacidad, costos y umbral con el área comercial.
- Medir una intervención con experimento controlado; las simulaciones económicas no son ahorro observado.
- Reejecutar auditorías por grupo y escenarios operativos cada vez que cambie el artefacto final.
