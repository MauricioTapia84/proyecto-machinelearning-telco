# Guía de defensa EP2 · Telco Customer Churn

## 1. Historia que debe contar el equipo

El proyecto responde una pregunta concreta: **¿podemos ordenar clientes según su riesgo histórico de abandono para ayudar a un equipo de retención a priorizar revisiones humanas?** No se intenta demostrar causalidad, automatizar una oferta ni afirmar que una campaña será rentable.

La historia completa tiene siete movimientos:

1. **Negocio:** perder un cliente en riesgo puede ser más costoso que revisar a un cliente que finalmente permanece; por eso se prioriza Recall.
2. **Datos:** el dataset tiene 7.043 registros, 19 predictores originales y una etiqueta binaria `Churn`; 1.869 registros son churn (26,54%).
3. **Evidencia exploratoria:** contratos mes a mes y menor antigüedad aparecen asociados a más churn. `tenure` y `TotalCharges` tienen correlación 0,826, pero esto es asociación, no causalidad.
4. **Evaluación válida:** se separan 5.617 filas de entrenamiento y 1.408 de test. La selección, la reducción de variables y el ajuste ocurren solo con train.
5. **Reducción:** las 46 columnas transformadas se filtran por correlación absoluta mayor que 0,95 y luego se aplica `SequentialFeatureSelector`, que conserva 10 columnas.
6. **Modelo:** se compara una referencia y varios clasificadores. La decisión final es un SVM lineal balanceado, ajustado con Optuna, sin SMOTE.
7. **Resultado y límite:** el modelo reducido mantiene las métricas de clasificación del modelo completo y pierde solo 0,0028 de ROC-AUC. Es un score de decisión, no una probabilidad calibrada.

La fuente numérica vigente es [README.md](../README.md), junto con [comparacion_variables_reducidas.csv](../output/resultados/comparacion_variables_reducidas.csv). Si una ejecución posterior cambia las cifras, se deben actualizar todos los documentos desde esos artefactos.

## 2. Datos, calidad y EDA

### Qué se revisó

- `TotalCharges` llegó como texto y contenía 11 espacios en blanco; todos corresponden a `tenure = 0`.
- `customerID` se excluye porque es un identificador, no un predictor generalizable.
- Se detectaron 22 perfiles repetidos al ignorar el identificador. Esto no prueba que sean la misma persona. La deduplicación se aplica solo en train y se revisó con sensibilidad.
- Las variables numéricas y categóricas se analizaron con estadísticas, distribuciones, boxplots y tasas de churn por categoría.

### Transformaciones coherentes con la distribución

La transformación se decide por tipo de variable y distribución; los parámetros se ajustan únicamente con entrenamiento:

| Variable o grupo | Evidencia | Tratamiento | Motivo |
|---|---|---|---|
| `TotalCharges` | Sesgo positivo alto, aproximadamente 0,962 | Conversión a numérico, imputación de respaldo, Yeo-Johnson y estandarización dentro del transformador | Reduce la cola derecha sin aplicar `StandardScaler` directamente al valor crudo. |
| `tenure` | Sesgo bajo, aproximadamente 0,253 | Imputación mediana y `StandardScaler` | Es una variable numérica con escala distinta, sin justificar una transformación de potencia. |
| `MonthlyCharges` | Sesgo leve, aproximadamente −0,227 | Imputación mediana y `StandardScaler` | La distribución no requiere una corrección fuerte de asimetría. |
| `SeniorCitizen` | Binaria, aunque llega como 0/1 | Imputación por moda y One-Hot Encoding | Se trata como categoría, no como una edad continua. |
| Categóricas nominales | No tienen orden numérico | Imputación por moda y One-Hot Encoding | Evita imponer una relación ordinal artificial. |

El EDA también construye variables derivadas para generar hipótesis, pero no las incorpora al modelo canónico. Los boxplots por `Churn` muestran separación descriptiva: sirven para interpretar y priorizar análisis, no para declarar por sí solos que una variable es predictora ni para probar causalidad. La figura principal es [boxplots_variables_seleccionadas_vs_churn.png](../output/graficos/boxplots_variables_seleccionadas_vs_churn.png).

Una limitación honesta: el EDA exploró la cohorte completa. La selección programática no consulta test, pero las decisiones humanas pudieron estar informadas por la exploración del dataset completo; por eso el test no debe presentarse como completamente ciego. La confirmación más fuerte sería una cohorte temporal o externa no explorada.

## 3. Contrato de train, test y fuga

El flujo canónico parte de los CSV procesados de [data/processed/](../data/processed/), no del CSV crudo durante el modelado:

```powershell
conda activate telco-ml
python scripts\generate_split.py
python scripts\evaluate_reduced_features.py
python scripts\run_project.py
```

`generate_split.py` crea el holdout por perfil, deduplica únicamente train y ajusta con train las operaciones necesarias para generar las matrices transformadas. El test se transforma con esos objetos ya ajustados, pero no se utiliza para seleccionar variables, comparar modelos ni ajustar hiperparámetros.

En el entrenamiento, [src/train.py](../src/train.py) carga `data/processed/train/`, filtra correlaciones, ejecuta SFS, selecciona el modelo y lo ajusta. Solo después de ese ajuste carga `data/processed/test/` para el reporte final. La misma regla se implementa en el notebook de modelado y en [evaluate_reduced_features.py](../scripts/evaluate_reduced_features.py).

No se debe decir que el modelo recibe el CSV crudo directamente. El artefacto canónico recibe la matriz transformada con las columnas que espera el SVM. La inferencia productiva requeriría encapsular también el preprocesamiento crudo como un contrato adicional.

## 4. Selección de variables y modelos

### Reducción de variables

La matriz inicial tiene 46 columnas transformadas. El filtro de correlación elimina 13 columnas redundantes, principalmente complementos de variables binarias One-Hot. Sobre las restantes se ejecuta SFS en dirección forward con cinco folds estratificados y objetivo fijo de 10 variables.

El estimador de screening es `LinearRegression`, con `neg_mean_squared_error`. Se usa para seleccionar predictores, no como clasificador final: `Churn` es binaria y la decisión final corresponde a un SVM.

Las 10 columnas seleccionadas son:

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

La selección se documenta en [seleccion_variables_sfs.csv](../output/resultados/seleccion_variables_sfs.csv) y el análisis de redundancia en [matriz_correlacion_train.csv](../output/resultados/matriz_correlacion_train.csv).

### Comparación y balanceo

Se comparan `DummyClassifier`, Regresión Logística, Random Forest, Gradient Boosting y SVM. La prioridad es Recall y ROC-AUC funciona como criterio complementario. No se utiliza SMOTE en el flujo final porque el desbalance se resuelve de forma suficiente y más simple mediante `class_weight='balanced'`; así se evita generar observaciones sintéticas en una matriz con variables One-Hot y no se altera el test.

El modelo final es el **SVM lineal balanceado ajustado por Optuna**, con `C≈0,1683`. Optuna solo explora configuraciones con validación cruzada de train. El artefacto persistido es [telco_churn_pipeline.joblib](../output/modelos/telco_churn_pipeline.joblib).

PCA y K-Means quedan como comparación exploratoria, no como parte de la decisión operativa: K-Means agrupa por similitud, no predice churn individual, y una separación débil no justifica usar segmentos como política comercial.

## 5. Resultados que sí deben presentarse

El test contiene 1.408 filas y 363 abandonos. La comparación reproducible es:

| Modelo | Variables | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM completo | 46 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8474 |
| SVM reducido | 10 | 0,7138 | 0,8430 | 0,4693 | 0,6030 | 0,8445 |

La matriz de confusión del modelo reducido es `[[699, 346], [57, 306]]`: detecta 306 de los 363 churn (Recall 84,30%), pero genera 346 falsos positivos. La conclusión correcta es **parquedad sin pérdida relevante en la decisión al umbral por defecto**, no “el modelo predice perfectamente”.

El ranking del SVM usa `decision_function`; no es una probabilidad calibrada. La captura del 20% superior es un indicador descriptivo de priorización histórica, no una tasa de retención ni una promesa de ahorro.

## 6. Preguntas de defensa y respuestas breves

**¿Por qué no usar `StandardScaler` directamente sobre `TotalCharges`?** Porque su cola derecha es marcada. Se usa Yeo-Johnson dentro del transformador y luego se estandariza la salida; así se reduce la asimetría sin filtrar información entre particiones.

**¿Por qué se usa `LinearRegression` si el objetivo es churn?** Solo como estimador de screening para SFS, tal como se solicitó. No es el clasificador final; el modelo final es SVM.

**¿Por qué no SMOTE?** El desbalance se maneja con pesos de clase. Es una alternativa más simple, evita ejemplos sintéticos y obtuvo el desempeño requerido sin tocar test. SMOTE solo sería justificable si una comparación reproducible con CV mostrara una mejora estable.

**¿Qué significa Recall 0,8430?** Que se detectaron 306 de los 363 abandonos reales del test al umbral del clasificador. Los 57 restantes son falsos negativos; los 346 falsos positivos muestran el costo de priorizar Recall.

**¿Se usó test para elegir variables?** No. SFS, el filtro de correlación, la comparación y Optuna usan únicamente train. Test se carga después del ajuste final.

**¿Qué demuestra el boxplot?** Una diferencia descriptiva de distribución entre clases. No demuestra causalidad ni sustituye validación cruzada.

**¿El modelo puede desplegarse ya?** No como campaña automática. Falta validar una cohorte nueva, calibrar o acordar un umbral operativo, comprobar costos y medir impacto con una prueba controlada.

## 7. Ensayo y evidencia

Para una defensa de 10 minutos con tres integrantes, cada persona puede cubrir tres diapositivas: problema y datos; EDA y evaluación; reducción, modelo y resultados. Todos deben poder responder preguntas sobre fuga, transformaciones, Recall, balanceo y limitaciones.

Antes de presentar, abrir:

- [README.md](../README.md), para objetivos, metodología y cifras.
- [Notebook EDA](../notebooks/01_EDA_y_Limpieza.ipynb), para calidad, distribuciones y boxplots.
- [Notebook de modelado](../notebooks/02_Modelado_y_Evaluacion.ipynb), para selección y evaluación.
- [modeling.py](../src/modeling.py), para correlación, SFS y elección del modelo.
- [comparacion_variables_reducidas.csv](../output/resultados/comparacion_variables_reducidas.csv), para los resultados finales.

La suite vigente contiene **13 pruebas automatizadas**. Esto verifica componentes del proyecto; no reemplaza una validación temporal, externa ni un piloto comercial.
