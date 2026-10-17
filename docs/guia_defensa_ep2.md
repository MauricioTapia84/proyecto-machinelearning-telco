# Guía exhaustiva de estudio y defensa · Telco Customer Churn

Esta guía explica el proyecto como un caso de aprendizaje y no solo como una lista de cifras. Sigue el ciclo de vida de un proyecto de analítica: traducir una necesidad de negocio a una pregunta medible, conocer los datos, preparar una evaluación válida, entrenar y comparar modelos, interpretar resultados y comunicar límites antes de proponer un uso.

La fuente de verdad para métricas vigentes es el [README](../README.md) y las salidas de los notebooks ejecutados. No memoricen cifras sin saber qué mide cada una. Si una nueva ejecución cambia resultados, actualicen los documentos y las diapositivas desde la evidencia nueva.

## 1. Resumen del caso en un minuto

La pregunta es: **¿podemos ordenar clientes de telecomunicaciones según su riesgo histórico de abandono para ayudar a un equipo a priorizar revisiones de retención?** Se trabaja con el dataset público Telco Customer Churn, que contiene 7.043 registros, 19 predictores originales y la etiqueta `Churn`. El churn histórico es 26,54%.

Se compararon Regresión Logística, Random Forest, Gradient Boosting y SVM, además de una representación experimental PCA + K-Means. El flujo seleccionado fue SVM con predictores originales. En validación cruzada de entrenamiento obtuvo Recall 0,8007 ± 0,0388 y ROC-AUC 0,8471 ± 0,0186; en el test reservado obtuvo Recall 0,7715, precisión 0,5116 y ROC-AUC 0,8405. El resultado es un **score de decisión**, no una probabilidad calibrada. El proyecto no demuestra que una campaña reduzca el churn ni que el escenario económico sea rentable.

## 2. Cómo pensar el proyecto de principio a fin

Antes de explicar un algoritmo, relaten la cadena de decisiones:

1. **Negocio:** se prioriza una cartera para revisión preventiva; no se pretende automatizar una oferta.
2. **Datos:** se identifica quién representa una fila, qué significa `Churn`, qué periodo describen los datos y qué problemas de calidad hay.
3. **Diseño de evaluación:** se aparta test y se decide qué se medirá en entrenamiento y validación.
4. **Preparación:** cada transformación que aprende algo de los datos se ajusta solo con el fold de entrenamiento correspondiente.
5. **Modelado:** se compara una línea base y varios modelos con métricas acordes al problema.
6. **Evaluación:** se reportan CV y test una vez cerrado el proceso de selección.
7. **Decisión:** se traducen errores y capacidad a un escenario explícito, sin confundir predicción con causalidad o rentabilidad.
8. **Entrega:** se guarda un pipeline reproducible, se documenta el entorno y se prepara la comunicación con límites y evidencia.

Esta secuencia ayuda a detectar errores comunes: elegir el modelo mirando el test, transformar toda la tabla antes de CV, presentar un score como probabilidad, confundir asociación con causa o decir que una simulación es un ahorro observado.

## 3. Explicación sección por sección

### 3.1 Problema, alcance y KPIs

**Qué se hace.** Se traduce “reducir abandono” a una tarea supervisada de clasificación binaria y a una decisión concreta: ordenar clientes para revisión.

**Por qué importa.** El objetivo define qué error es más preocupante y qué métricas sirven. Un falso negativo es un cliente que abandonó y el modelo no marcó; un falso positivo es un cliente marcado que no abandonó. Como la meta exploratoria es detectar churn, Recall es importante, pero elevar Recall suele aumentar los falsos positivos y el costo de contactar.

**Qué significa el KPI.** La tasa histórica de churn de 26,54% describe este dataset, no una tasa futura garantizada. Recall mide la fracción de abandonos reales detectada a un umbral específico. ROC-AUC evalúa ordenamiento a través de muchos umbrales y no decide por sí sola cuántos clientes contactar. La captura del 20% superior es un ranking bajo una capacidad hipotética; no es la eficacia de una campaña.

**Herramientas/evidencia.** Definición y fórmulas en el [README](../README.md), celdas iniciales del [notebook de modelado](../notebooks/02_Modelado_y_Evaluacion.ipynb) y `scripts/calculate_business_kpis.py`.

**Qué decir.** “El modelo apoya una priorización para revisión humana. No decide automáticamente la intervención y el score de SVM no está calibrado como probabilidad.”

### 3.2 Fuente, unidad de análisis y calidad

**Qué se hace.** Se inspeccionan estructura, tipos, valores ausentes, duplicados, etiquetas y consistencia de cada registro.

**Hallazgos.** Hay 7.043 filas y 21 columnas de origen, incluida la etiqueta. `TotalCharges` llegó como texto y 11 valores eran espacios en blanco; estos corresponden a `tenure = 0`. Se convierten los cargos a numérico y se documenta el tratamiento. La etiqueta tiene 1.869 abandonos y 5.174 permanencias.

**Duplicados/perfiles repetidos.** No hay filas idénticas incluyendo `customerID`. Tras normalizar campos, se detectan 22 perfiles repetidos si se ignora el identificador e incluye `Churn`. Esto no prueba que sean la misma persona: se hizo una comparación de sensibilidad y la política de deduplicación evita dar peso repetido a perfiles iguales. No debe describirse como identificación de clientes.

**Por qué importa.** Si el tipo o significado de una columna se interpreta mal, el pipeline aprende sobre datos defectuosos. Si copias de un perfil llegan tanto a entrenamiento como a validación, el rendimiento puede parecer artificialmente bueno.

**Herramientas/evidencia.** El [notebook EDA](../notebooks/01_EDA_y_Limpieza.ipynb), `src/data.py` y la comparación de sensibilidad documentada en el README.

**Qué decir.** “Revisamos el significado y la calidad antes de modelar; corregimos `TotalCharges` y tratamos los perfiles repetidos con cautela, sin afirmar que sean clientes idénticos.”

### 3.3 EDA: distribuciones, relación con churn y correlación

**Qué se hace.** Se describen distribuciones y se comparan variables con churn usando tablas y gráficos univariados, bivariados y multivariados.

**Hallazgos que se pueden comunicar.** El churn histórico es mayor en contratos mes a mes y entre clientes con menor antigüedad. También hay diferencias descriptivas por servicios, métodos de pago y cargos. La relación entre `tenure` y `TotalCharges` es alta (correlación 0,826), lo que es coherente con que el total facturado crezca con el tiempo de permanencia.

**Cómo interpretar la correlación.** Una correlación resume una asociación numérica, no demuestra causa. Correlación alta entre dos predictores tampoco obliga a eliminar una variable: se debe considerar el propósito, la representación del modelo y la validación. No se ejecutó una regla automática de eliminación por correlación >0,95 ni se debe insinuar que se hizo.

**Límite importante.** El EDA inspeccionó la cohorte completa, incluida la etiqueta del conjunto que más tarde se reservó como test. Aunque la selección programática no consulta el test antes del reporte, la exploración pudo influir en decisiones humanas. Por eso el test no es completamente ciego; se necesita cohorte temporal o externa para confirmación independiente.

**Población de aplicación.** La evidencia corresponde a la población histórica representada por este dataset público de clientes Telco. No se ha demostrado que el modelo sea válido para otra compañía, país, producto, periodo o definición de churn; antes de trasladarlo habría que comprobar que variables, población, calidad y prevalencia sean comparables.

**Herramientas/evidencia.** Gráficos en `output/graficos/` y análisis en el [notebook EDA](../notebooks/01_EDA_y_Limpieza.ipynb).

**Qué decir.** “El EDA genera hipótesis y explica patrones de esta muestra; no establece causalidad ni reemplaza la evaluación en datos futuros.”

### 3.4 Partición y prevención de fuga

**Qué se hace.** Se mantiene una partición estratificada de train y test para que la proporción de churn sea comparable. La selección de modelos y parámetros ocurre dentro de entrenamiento mediante validación cruzada de cinco folds.

**Qué es fuga de información.** Ocurre cuando información que no estaría disponible al entrenar —por ejemplo, estadísticas calculadas sobre todo el dataset o ejemplos del fold de validación— influye en el ajuste. Esto puede hacer que la evaluación parezca mejor de lo que será en datos nuevos.

**Cómo se evita en el flujo canónico.** Imputación, escalado, codificación, PCA/K-Means en la rama correspondiente y SMOTE están dentro del pipeline de evaluación. En cada fold, el pipeline aprende esas operaciones solo a partir de la partición de entrenamiento de ese fold y después transforma su validación. El test se reserva para el reporte final, no para elegir el modelo o ajustar hiperparámetros.

**Límite.** Esto evita la fuga programática en el proceso descrito, pero no vuelve ciego el test ante la exploración humana previa del EDA.

**Herramientas/evidencia.** `src/pipeline.py`, `src/modeling.py` y `notebooks/02_Modelado_y_Evaluacion.ipynb`.

### 3.5 Preparación de variables y SMOTE

**Qué se hace.** Se separan variables numéricas y categóricas, se imputan los valores necesarios, se escalan numéricas y se codifican categorías. El flujo conserva las transformaciones junto con el estimador.

**SMOTE.** La clase minoritaria se sobremuestrea dentro del pipeline y solo en los datos de entrenamiento de cada fold. No se aplica al test. De lo contrario, la validación evaluaría ejemplos sintéticos o tendría una distribución artificial.

**Por qué se usa un pipeline.** Garantiza que el mismo preprocesamiento se ajuste en entrenamiento, se aplique a validación/test y quede guardado con el modelo para inferencia; reduce diferencias entre notebook, entrenamiento y uso posterior.

### 3.6 Modelos supervisados y selección

**Qué se compara.** DummyClassifier como referencia base, Regresión Logística, Random Forest, Gradient Boosting y Support Vector Machine (SVM), principalmente con predictores originales y una variante PCA + K-Means.

**Por qué varios modelos.** Los algoritmos tienen supuestos distintos. La comparación estima cuál es más útil bajo la métrica y el procedimiento definidos; un algoritmo conocido no se elige por prestigio.

**Selección actual.** El SVM ajustado con Optuna y predictores originales se seleccionó mediante Recall en CV de entrenamiento; ROC-AUC se reportó como criterio adicional. Optuna hizo una búsqueda acotada. La evaluación del candidato se mantuvo dentro de entrenamiento antes del reporte del test.

**PCA y K-Means.** PCA reduce dimensiones y K-Means asigna grupos por similitud no supervisada; ninguno conoce la etiqueta por definición del algoritmo. En esta comparación, silhouette fue 0,0462 y el SVM con PCA + K-Means obtuvo Recall CV 0,7751, por debajo de 0,8007 con predictores originales. Se conserva como comparación exploratoria, no como herramienta de decisión comercial.

**Herramientas/evidencia.** `src/modeling.py`, `src/pipeline.py` y el [notebook de modelado](../notebooks/02_Modelado_y_Evaluacion.ipynb).

### 3.7 Métricas y resultados

| Medida | Resultado vigente | Interpretación |
|---|---:|---|
| Recall CV | 0,8007 ± 0,0388 | En promedio, identifica cerca del 80% de los churn en los folds de entrenamiento; la desviación muestra variación entre folds. |
| ROC-AUC CV | 0,8471 ± 0,0186 | Capacidad de ordenar positivos por encima de negativos a través de umbrales. |
| Recall test | 0,7715 (287/372) | Con el umbral de clasificación usado en el reporte, detectó 287 de 372 churn del test. |
| Precisión test | 0,5116 (287/561) | De los clientes marcados como churn, alrededor de 51% eran churn en ese test. |
| F1 test | 0,62 | Equilibrio armónico entre precisión y Recall en ese punto de operación. |
| ROC-AUC test | 0,8405 | Ordenamiento medido sobre el test final; no se usa para cambiar el modelo. |
| Captura en 20% superior | 192/372 = 51,61% | Un ranking limitado a 281 clientes concentra cerca de la mitad de los churn del test; es capacidad ilustrativa, no eficacia de retención. |

**Matrices y errores.** La matriz de confusión del umbral por defecto reporta 759 TN, 274 FP, 85 FN y 287 TP. No confundan ese punto de operación con el escenario económico posterior: el umbral OOF de ese escenario es distinto y selecciona 44 de 1.405 clientes de test, no los 281 del ranking top 20%.

**Por qué no basta una métrica.** Un modelo puede tener alto Recall y muchos falsos positivos. La métrica correcta depende de costos, capacidad y objetivo operativo; esos valores comerciales aún no están disponibles.

### 3.8 Importancia de variables y explicaciones

La permutación se ejecutó en un holdout explicativo de entrenamiento, no en test ni para volver a seleccionar el ganador. Indica cuánto cambia una métrica al desordenar una variable en esa muestra y bajo ese modelo; no prueba que la variable cause churn. Las explicaciones SHAP/waterfall actuales describen el score del modelo nuevo, y deben leerse como contribución local/global al modelo, no como efecto causal ni recomendación automática.

### 3.9 Errores por grupos y cautela ética

La auditoría descriptiva incluye género, `SeniorCitizen`, contrato, internet y antigüedad, con conteos e intervalos Wilson de 95%. Algunos resultados a conocer:

- `SeniorCitizen=1`: Recall 0,910 y FPR 0,493; `SeniorCitizen=0`: Recall 0,728 y FPR 0,230.
- Contrato mes a mes: Recall 0,864 y FPR 0,600.
- Contrato anual: Recall 0,059.
- Contrato de dos años: Recall 0 con solo 8 churn; esa estimación es especialmente incierta.

Estas diferencias justifican investigar errores, soporte y consecuencias; no bastan para certificar equidad ni demostrar sesgo. Es una evaluación post hoc en el test ya explorado por el EDA y no se utilizó para retocar el modelo. Si el proyecto avanzara, habría que acordar métricas de equidad, revisar variables/procesos y volver a medir en datos apropiados.

### 3.10 Escenario económico hipotético

**Supuestos:** capacidad de 250 contactos por cada 7.043 clientes, horizonte de 24 meses, eficacia hipotética de retención del 20%, costo de contacto igual a 1%, 5% o 10% de la mediana de `MonthlyCharges` (70,4 unidades) y valor incremental retenido igual a cero.

**Cálculo conceptual:** pérdida base proxy = churn observado × 24 × mediana de cargos mensuales. Pérdida bruta supuestamente evitada = verdaderos positivos contactados × eficacia supuesta × proxy por cliente. Se agregan costo de contacto y pérdida residual.

En el test de este escenario se priorizaron 44 clientes, hubo 40 TP y el Recall del escenario fue 0,1075; con eficacia hipotética del 20% se estiman 8 retenciones, solo como ejercicio aritmético. El ahorro proxy resultó aproximadamente 13.207–13.486 unidades según costo de contacto.

**Lo que no significa:** no son ahorros observados, valor de vida del cliente, margen ni utilidad. `MonthlyCharges` representa cargos brutos, la moneda no está especificada y no se hizo campaña. Tampoco se midió causalidad con control. La decisión requiere datos comerciales y un piloto controlado. El artefacto conserva `operating_threshold=None`.

### 3.11 Artefacto, inferencia y reproducibilidad

El flujo oficial se ejecuta con `python scripts\run_project.py`. El pipeline completo y metadata quedan en `output/modelos/telco_churn_pipeline.joblib`; `src/inference.py` valida entradas y produce predicción/score sin requerir probabilidades. Gráficos, modelos y resultados se guardan en `output/`.

El entorno está declarado en `environment.yml` y el lock explícito Windows x64 en `environment-win-64.lock.txt`. La suite vigente tiene 12 pruebas. Esto prueba componentes definidos en las pruebas, no equivale a validación en producción o cohorte externa.

Comandos de comprobación:

```powershell
conda activate telco-ml
python -m unittest discover -s tests -v
python scripts\evaluate_operating_scenarios.py
python scripts\evaluate_group_metrics.py
```

La generación completa del modelo ejecuta la búsqueda Optuna y tarda más que las pruebas.

## 4. Preguntas de defensa para estudiar

Practiquen primero una respuesta breve, luego una respuesta técnica. No basta con recitar:

1. ¿Qué decisión apoya el modelo y qué no hace?
2. ¿Por qué el churn se trata como clasificación binaria?
3. ¿Cuál es la unidad de análisis y qué problemas se encontraron en los datos?
4. ¿Qué significa que existan perfiles repetidos y por qué no prueba identidad?
5. ¿Qué aporta la correlación 0,826 entre `tenure` y `TotalCharges`? ¿Qué no permite concluir?
6. ¿Cómo se evita fuga al ajustar imputación, escalado, codificación y SMOTE?
7. ¿Cuál es la diferencia entre train, validación cruzada y test?
8. ¿Qué significa Recall 0,7715? ¿Qué significan los 85 FN y 274 FP?
9. ¿Por qué se usa ROC-AUC junto con Recall?
10. ¿El score del SVM es una probabilidad calibrada? ¿Cómo se sabe?
11. ¿Por qué se deja K-Means fuera de la decisión operativa?
12. ¿Qué justifica el umbral del escenario y por qué no está dentro del artefacto?
13. ¿Por qué el ahorro del escenario no puede presentarse como resultado financiero?
14. ¿Qué muestran las métricas de errores por grupo y qué no demuestran?
15. ¿Qué limitación deja el EDA respecto de la independencia del test?
16. ¿Qué evidencia ofrece el pipeline guardado y qué falta para una aplicación productiva?
17. ¿Qué experimento permitiría medir si una campaña evita churn?
18. Si el profesor cuestiona el rendimiento, ¿qué revisión harían sin tocar el test actual?

## 5. Rutina de ensayo para tres integrantes

1. Cada integrante estudia el proyecto completo; el reparto de exposición no limita las preguntas individuales.
2. Cada uno explica su tercio sin leer y contesta al azar dos preguntas de la sección 4.
3. Un compañero verifica las cifras contra README/notebooks y apunta expresiones que confundan score, probabilidad, captura, retención, asociación o causalidad.
4. Ensayen con cronómetro: 10 minutos total; cada tercio dispone de unos 3 minutos y 20 segundos, incluyendo su transición.
5. Abran de antemano evidencia local; no dependan de Internet, de un kernel ejecutándose ni de reproducir Optuna en vivo.
6. Registren la práctica real por persona. La existencia de esta guía no equivale a demostrar que el grupo practicó.

| Integrante | Ensayo realizado | Tiempo | Preguntas correctas / asignadas | Próximo tema |
|---|---|---:|---:|---|
| 1 | ☐ | ____ | ____ / ____ | ____ |
| 2 | ☐ | ____ | ____ / ____ | ____ |
| 3 | ☐ | ____ | ____ / ____ | ____ |

## 6. Evidencia que deben poder abrir

- [README del proyecto](../README.md): problema, KPIs, metodología y cifras.
- [Notebook EDA](../notebooks/01_EDA_y_Limpieza.ipynb): calidad, distribuciones, correlación y conclusiones.
- [Notebook de modelado](../notebooks/02_Modelado_y_Evaluacion.ipynb): partición, CV, comparación, explicabilidad, test y limitaciones.
- `src/pipeline.py`, `src/modeling.py`, `src/decision.py` y `src/inference.py`: preparación, evaluación, escenario de umbral e inferencia.
- `output/graficos/` y `output/resultados/`: figuras y tablas citadas.
- `environment.yml`, `environment-win-64.lock.txt` y `tests/`: entorno y comprobaciones reproducibles.
