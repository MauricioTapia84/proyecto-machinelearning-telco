# Telco Customer Churn · EP2 Machine Learning

Proyecto de clasificación supervisada para ordenar clientes según su riesgo histórico de abandono y apoyar la priorización de revisiones de retención. El modelo no automatiza campañas, no demuestra causalidad y el score del SVM no es una probabilidad calibrada.

## 1. Objetivo y resultado principal

La pregunta de negocio es: **¿podemos priorizar clientes con mayor riesgo histórico de churn cuando la capacidad de revisión es limitada?** Se prioriza Recall porque un falso negativo puede ser más costoso que revisar un cliente que finalmente permanece; la relación de costos aún no está cuantificada comercialmente.

El flujo vigente selecciona un **SVM lineal balanceado**, ajustado por Optuna, sin SMOTE. Parte de 46 columnas transformadas, elimina redundancias con un filtro de correlación absoluta mayor que 0,95 y aplica `SequentialFeatureSelector` para conservar 10 columnas. `LinearRegression` se usa únicamente como estimador de screening para SFS.

Resultado actual sobre test:

| Modelo | Variables | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM completo | 46 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8474 |
| SVM reducido | 10 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8445 |

El test tiene 1.408 filas y 363 abandonos. La matriz de confusión del modelo reducido es `[[699, 346], [57, 306]]`; detecta 306 churn y deja 57 falsos negativos. La reducción conserva las métricas de clasificación y pierde 0,0028 de ROC-AUC.

## 2. Datos y calidad

Se utiliza el dataset público Telco Customer Churn, con 7.043 registros, 19 predictores originales y `Churn` como objetivo binario. Hay 1.869 abandonos (26,54%). `TotalCharges` llegó como texto y contiene 11 espacios en blanco, asociados a `tenure = 0`. Se excluye `customerID` por ser un identificador. Se encontraron 22 perfiles repetidos ignorando el identificador; la deduplicación se aplica solo en train y no implica que los registros sean necesariamente la misma persona.

El EDA muestra mayor churn descriptivo en contratos mes a mes y clientes de menor antigüedad. `tenure` y `TotalCharges` presentan correlación 0,826. Estas relaciones generan hipótesis, pero no prueban causalidad. Los boxplots por `Churn` son evidencia descriptiva y no una regla automática de selección.

## 3. Transformaciones según distribución

| Variable o grupo | Tratamiento | Justificación |
|---|---|---|
| `TotalCharges` | Conversión a numérico, imputación de respaldo, Yeo-Johnson y estandarización | Sesgo derecho alto; no se aplica `StandardScaler` directamente al valor crudo. |
| `tenure` | Imputación mediana y `StandardScaler` | Sesgo bajo y escala numérica distinta. |
| `MonthlyCharges` | Imputación mediana y `StandardScaler` | Sesgo leve; se requiere principalmente escala comparable. |
| `SeniorCitizen` | Moda y One-Hot Encoding | Es una categoría binaria, no una magnitud continua. |
| Categóricas nominales | Moda y One-Hot Encoding | Evita imponer orden artificial. |

Los parámetros se ajustan con entrenamiento. El CSV crudo solo se usa para generar las matrices procesadas; el modelado consume `data/processed/`.

## 4. Contrato de train/test y fuga

La generación de datos canónica es:

```powershell
conda activate telco-ml
python scripts\generate_split.py
```

El holdout se genera por perfil: train tiene 5.617 filas y test 1.408. La deduplicación ocurre únicamente en train. Las transformaciones se ajustan con train y se aplican a test sin volver a aprender parámetros sobre test.

Durante la selección y el entrenamiento, [src/train.py](src/train.py), [src/modeling.py](src/modeling.py) y [scripts/evaluate_reduced_features.py](scripts/evaluate_reduced_features.py) cargan solo `data/processed/train/`. La matriz procesada de test se carga después del ajuste final para generar el reporte. Test no interviene en el filtro de correlación, SFS, comparación de modelos ni Optuna.

## 5. Flujo completo

```powershell
conda activate telco-ml
python scripts\generate_split.py
python scripts\evaluate_reduced_features.py
python scripts\run_project.py
```

- [src/processed_data.py](src/processed_data.py): partición, deduplicación y matrices procesadas.
- [src/pipeline.py](src/pipeline.py): transformaciones por distribución.
- [src/modeling.py](src/modeling.py): correlación, SFS, comparación y selección.
- [src/train.py](src/train.py): entrenamiento y evaluación final.
- [scripts/evaluate_reduced_features.py](scripts/evaluate_reduced_features.py): evidencia de reducción.
- [scripts/run_project.py](scripts/run_project.py): entrada oficial para persistir el modelo.
- [output/modelos/telco_churn_pipeline.joblib](output/modelos/telco_churn_pipeline.joblib): artefacto actual.

El artefacto actual espera la matriz transformada con las columnas seleccionadas; no se debe enviar el CSV crudo directamente. Para producción habría que encapsular y validar también el preprocesamiento de entrada.

## 6. Variables seleccionadas

```text
power__TotalCharges
standard__tenure
categorical__MultipleLines_No
categorical__InternetService_Fiber optic
categorical__InternetService_No
categorical__StreamingTV_No
categorical__StreamingMovies_No
categorical__Contract_Month-to-month
categorical__PaperlessBilling_No
categorical__PaymentMethod_Electronic check
```

La matriz de correlación está en [output/resultados/matriz_correlacion_train.csv](output/resultados/matriz_correlacion_train.csv), la selección en [output/resultados/seleccion_variables_sfs.csv](output/resultados/seleccion_variables_sfs.csv) y la comparación completa/reducida en [output/resultados/comparacion_variables_reducidas.csv](output/resultados/comparacion_variables_reducidas.csv). El gráfico de boxplots es [output/graficos/boxplots_variables_seleccionadas_vs_churn.png](output/graficos/boxplots_variables_seleccionadas_vs_churn.png).

## 7. Interpretación y límites

Recall 0,8430 significa que el modelo detectó 306 de los 363 abandonos del test al umbral por defecto. La precisión 0,4693 y los 346 falsos positivos muestran el costo de priorizar la detección. El score es `decision_function`, no probabilidad calibrada.

PCA y K-Means quedan como comparación exploratoria. K-Means no predice churn individual y no define una política de contacto. Las auditorías por grupo son descriptivas y no certifican equidad. El siguiente paso es validar en una cohorte temporal o externa y medir una intervención mediante una prueba controlada.

## 8. Documentación y reproducibilidad

- [docs/guia_defensa_ep2.md](docs/guia_defensa_ep2.md): guía técnica y preguntas de defensa.
- [docs/presentacion_ep2.md](docs/presentacion_ep2.md): guion de storytelling para 9 diapositivas.
- [docs/revision_integral_ep2.md](docs/revision_integral_ep2.md): confirmación de coherencia entre etapas.
- [docs/checklist.md](docs/checklist.md): requisitos y pendientes.
- [docs/estructura_proyecto.md](docs/estructura_proyecto.md): organización de carpetas.
- [notebooks/01_EDA_y_Limpieza.ipynb](notebooks/01_EDA_y_Limpieza.ipynb): EDA y limpieza.
- [notebooks/02_Modelado_y_Evaluacion.ipynb](notebooks/02_Modelado_y_Evaluacion.ipynb): selección y evaluación.

El entorno oficial se declara en [environment.yml](environment.yml) y la resolución Windows x64 en [environment-win-64.lock.txt](environment-win-64.lock.txt). La suite vigente contiene 13 pruebas automatizadas; esto no reemplaza validación externa ni un piloto comercial.
