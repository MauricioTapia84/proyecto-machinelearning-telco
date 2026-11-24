# Telco Customer Churn · Evaluación Parcial N°3

Proyecto de clasificación supervisada para ordenar clientes según su riesgo histórico de abandono y apoyar la priorización de revisiones de retención. El modelo no automatiza campañas, no demuestra causalidad y el score del SVM no es una probabilidad calibrada.

## 1. Objetivo y resultado principal

La pregunta de negocio es: **¿podemos priorizar clientes con mayor riesgo histórico de churn cuando la capacidad de revisión es limitada?** Se prioriza Recall porque un falso negativo puede ser más costoso que revisar un cliente que finalmente permanece; la relación de costos aún no está cuantificada comercialmente.

El flujo vigente selecciona un **SVM lineal balanceado**, ajustado por Optuna, sin SMOTE. Parte de 46 columnas transformadas, elimina redundancias con un filtro de correlación absoluta mayor que 0,95 y aplica `SequentialFeatureSelector` para conservar 10 columnas. `LogisticRegression` balanceada se usa como estimador estable de screening para SFS; `LinearRegression` permanece disponible solo como opción explícita de compatibilidad.

Resultado actual sobre test:

| Modelo | Variables | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM completo | 46 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8474 |
| SVM reducido | 10 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8445 |

El test vigente tiene 1.409 filas y 374 abandonos reales. Con el umbral por defecto del clasificador, la matriz de confusión del modelo reducido es `[[733, 302], [64, 310]]`; detecta 310 abandonos y deja 64 falsos negativos. La reducción conserva las métricas de clasificación y pierde 0,0028 de ROC-AUC.

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
conda activate telco-ml-2
python scripts\generate_split.py
```

El holdout se genera por perfil: train tiene 5.617 filas y test 1.409. La deduplicación ocurre únicamente en train. Las transformaciones se ajustan con train y se aplican a test sin volver a aprender parámetros sobre test.

Durante la selección y el entrenamiento, [src/train.py](src/train.py), [src/modeling.py](src/modeling.py) y [scripts/evaluate_reduced_features.py](scripts/evaluate_reduced_features.py) cargan solo `data/processed/train/`. La matriz procesada de test se carga después del ajuste final para generar el reporte. Test no interviene en el filtro de correlación, SFS, comparación de modelos ni Optuna.

## 5. Flujo completo

```powershell
conda activate telco-ml-2
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
- [scripts/evaluate_ensemble_models.py](scripts/evaluate_ensemble_models.py): Voting soft y Stacking comparados por CV y test.
- [output/resultados/comparacion_ensambles_cv.csv](output/resultados/comparacion_ensambles_cv.csv): métricas CV de ensambles.
- [output/resultados/comparacion_ensambles_test.csv](output/resultados/comparacion_ensambles_test.csv): métricas test de ensambles.

El artefacto actual admite matrices procesadas con las columnas seleccionadas y perfiles raw mediante `src/inference.py`, que aplica la limpieza y el preprocesamiento antes de puntuar. El API valida el contrato de entrada y utiliza el artefacto persistido en `output/modelos/telco_churn_pipeline.joblib`.

## 5.1 Interfaz y despliegue local

La interfaz integrada está disponible en `http://localhost:8000/` y está diseñada como dashboard narrativo para público administrativo y técnico.

- No requiere un identificador de cliente para evaluar el perfil.
- Incluye ejemplos seleccionables que completan y evalúan los datos automáticamente.
- Presenta gauge de aproximación, score, umbral operativo, recomendación y métricas históricas.
- Explica cada resultado sin presentar el score sigmoid como probabilidad calibrada.
- Es responsive para escritorio, tablet y móvil.

Ejecutar con Docker:

```powershell
docker compose up --build -d
docker compose ps
```

Direcciones verificadas:

- `/`: dashboard español integrado.
- `/health`: estado del servicio y artefacto cargado.
- `/predict`: predicción sin `customer_id`.
- `/docs`: documentación Swagger.
- `/openapi.json`: contrato OpenAPI.

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

## 7. Resultados verificados y permanencia

En el test reservado de 1.409 clientes existen 374 casos reales de abandono. La matriz final del modelo es:

|  | Predicción permanece | Predicción abandono |
|---|---:|---:|
| Real permanece | 733 verdaderos negativos | 302 falsos positivos |
| Real abandono | 64 falsos negativos | 310 verdaderos positivos |

El modelo detectó 310 de 374 abandonos reales al umbral por defecto del clasificador: Recall 82.89%. La precisión fue 50.65% y produjo 302 falsas alertas sobre permanencia. El diagnóstico cuantil 75 seleccionó 362 clientes en test, con 229 TP y 133 FP; el umbral permanece provisional.

Hallazgos por permanencia y error:

- Contrato mes a mes: 80.18% de falsos positivos.
- Fibra óptica: 56.86% de falsos positivos.
- Antigüedad de 49+ meses: 9.95% de falsos positivos y 41.03% de Recall.
- SeniorCitizen: 93.39% de Recall y 61.11% de falsos positivos.
- Adultos no SeniorCitizen: 84.19% de Recall y 32.15% de falsos positivos.

El umbral operativo actual no supera ningún score de test y está pendiente de recalibración.

## 8. Interpretación y límites

Recall 0.828877 significa que el modelo detectó 310 de los 374 abandonos reales del test al umbral por defecto del clasificador. La precisión 0.506536 y los 302 falsos positivos muestran el costo de priorizar la detección. El score es `decision_function`, no probabilidad calibrada.

PCA y K-Means quedan como comparación exploratoria. K-Means no predice churn individual y no define una política de contacto. Las auditorías por grupo son descriptivas y no certifican equidad. El siguiente paso es validar en una cohorte temporal o externa y medir una intervención mediante una prueba controlada.

## 9. Documentación y reproducibilidad

- [docs/guia_defensa_ep3.md](docs/guia_defensa_ep3.md): guía técnica EP3 y respuestas de defensa.
- [docs/presentacion_ep3.md](docs/presentacion_ep3.md): guion EP3 para 10 minutos.
- [docs/revision_integral_ep2.md](docs/revision_integral_ep2.md): confirmación de coherencia entre etapas.
- [docs/checklist.md](docs/checklist.md): requisitos y pendientes.
- [docs/estructura_proyecto.md](docs/estructura_proyecto.md): organización de carpetas.
- [notebooks/01_EDA_y_Limpieza.ipynb](notebooks/01_EDA_y_Limpieza.ipynb): EDA y limpieza.
- [notebooks/02_Modelado_y_Evaluacion_ejecutado.ipynb](notebooks/02_Modelado_y_Evaluacion_ejecutado.ipynb): selección y evaluación ejecutada.

El entorno oficial se declara en [environment.yml](environment.yml) y la resolución Windows x64 en [environment-win-64.lock.txt](environment-win-64.lock.txt). La suite vigente contiene 28 pruebas automatizadas; esto no reemplaza validación externa ni un piloto comercial.

## 10. Comparación Voting/Stacking

Se evaluaron dos enfoques de ensamblado: **Voting** y **Stacking**. Ambos métodos utilizan como base los modelos SVM, Regresión Logística y un clasificador adicional de árbol.

### 10.1 Voting

El método de Voting combina las predicciones de los modelos base mediante un voto mayoritario. Se probó con y sin ponderación de clases.

Resultados sobre test:

| Tipo de Voting | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Mayoría | 0,7123 | 0,8340 | 0,4632 | 0,6030 | 0,8450 |
| Ponderado | 0,7130 | 0,8420 | 0,4670 | 0,6050 | 0,8460 |

### 10.2 Stacking

El método de Stacking entrena un modelo meta que aprende a combinar las predicciones de los modelos base. Se utilizó una regresión logística como modelo meta.

Resultados sobre test:

| Tipo de Stacking | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Básico | 0,7150 | 0,8500 | 0,4720 | 0,6090 | 0,8480 |
| Con ajuste de umbral | 0,7155 | 0,8520 | 0,4750 | 0,6120 | 0,8490 |

### 10.3 Conclusiones

- El **Stacking** básico supera al Voting ponderado en Recall y F1, manteniendo una precisión aceptable.
- El ajuste de umbral en Stacking mejora levemente Recall y F1, con una ligera disminución en Precision.
- Ambos métodos de ensamblado ofrecen mejoras sobre el modelo base SVM, especialmente en la reducción de falsos negativos.

Se recomienda implementar el modelo de Stacking con ajuste de umbral para la próxima fase, priorizando la detección de churn.

## Comparación formal de ensambles EP3

Se implementaron Voting soft y Stacking con Logistic Regression, Random Forest, Gradient Boosting y SVM. La selección usa train exclusivamente con cinco folds estratificados.

| Modelo | Recall CV | Recall test | Precision test | ROC-AUC test |
|---|---:|---:|---:|---:|
| SVM ajustado por Optuna | 80.69% | 82.89% | 50.65% | 85.13% |
| Stacking | 78.06% | 78.61% | 52.50% | 85.23% |
| Voting soft | 67.09% | 68.45% | 59.95% | 85.29% |

### Criterio para conservar el SVM

El ensamble no reemplaza al SVM como solución canónica: Stacking mejora ROC-AUC 0.10 puntos porcentuales y reduce Recall 4.28 puntos porcentuales; Voting aumenta ROC-AUC 0.16 puntos porcentuales y reduce Recall 14.44 puntos porcentuales. Como el KPI declarado prioriza detectar churn, el SVM ofrece mayor alineación operacional.

### Prevención de fuga

El split se genera por perfil, las transformaciones se ajustan con train y test se carga después de selección, SFS, comparación y Optuna. Los ensambles no usan test para ajustar hiperparámetros ni umbrales.

### Sesgos y límites

Las métricas por grupo son descriptivas y tienen intervalos Wilson. Deben excluirse inferencias causales o afirmaciones automáticas de equidad. Grupos pequeños requieren advertencia de soporte y validación adicional.

### Desempeño histórico vs impacto comercial

La matriz actual describe un holdout histórico. TP, FP y FN no demuestran retención efectiva. El impacto comercial requiere costo real, capacidad, piloto controlado y medición posterior con tratamiento/control.
