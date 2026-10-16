# Guía de defensa EP2 · Telco Customer Churn

Esta guía sirve para que cada integrante pueda explicar cualquier parte del proyecto. Antes de la entrega, cada persona debería responder oralmente las preguntas sin leer las respuestas.

## 1. Problema y objetivo

**¿Qué problema se aborda?**

Priorizar clientes con mayor riesgo de abandonar el servicio para apoyar revisiones preventivas de retención.

**¿Qué entrega el modelo?**

Una probabilidad o ranking de riesgo de churn. No ejecuta campañas ni toma decisiones automáticas sobre los clientes.

## 2. Datos y calidad

**¿Cuál es la fuente y cuál es la variable objetivo?**

El dataset público Telco Customer Churn, con `Churn` como variable objetivo binaria (`Yes`/`No`). Contiene 7.043 clientes, 19 predictores originales y la variable objetivo.

**¿Qué ocurrió con `TotalCharges`?**

Llegó como texto y contenía espacios vacíos. Se convirtió a numérico; los casos con `tenure = 0` se imputaron con 0 y cualquier caso restante se trató con la mediana correspondiente.

**¿Cómo se evitó la fuga de información?**

Se separó train/test con estratificación. Las transformaciones estadísticas y el balanceo se ajustan dentro del entrenamiento; el test se reserva para la evaluación final.

## 3. EDA y preparación

**¿Cuál fue el principal hallazgo del EDA?**

La tasa histórica de churn es 26,54%. El abandono es mayor en contratos mes a mes y entre clientes con menor antigüedad; también se observan diferencias descriptivas por servicios, método de pago y cargos.

**¿Esos hallazgos prueban causalidad?**

No. Son asociaciones observadas en el dataset y sirven para explorar y orientar la priorización, pero no demuestran que una variable cause el abandono.

## 4. Aprendizaje no supervisado

**¿Qué se hizo con PCA y K-Means?**

Se evaluó una representación con PCA y tres segmentos K-Means. Los clusters se interpretaron como contexto descriptivo, no como reemplazo del modelo de riesgo.

**¿Por qué no se eligió PCA + K-Means como representación final?**

La representación original obtuvo mejor Recall para el objetivo priorizado de detectar churn, por lo que se mantuvo como referencia final.

## 5. Modelamiento y selección

**¿Qué modelos se compararon?**

Regresión Logística, Random Forest, Gradient Boosting y Support Vector Machine, usando tanto predictores originales como la representación PCA + K-Means.

**¿Por qué se priorizó Recall?**

Porque el objetivo es detectar clientes que efectivamente abandonan. Un falso negativo representa un cliente en riesgo que no sería priorizado para revisión.

**¿Cómo se trató el desbalance?**

Con SMOTE dentro del pipeline de entrenamiento. El conjunto de prueba no se modifica con SMOTE.

**¿Cuál fue el modelo seleccionado y cómo?**

SVM con predictores originales. Se seleccionó mediante Recall promedio de validación cruzada sobre entrenamiento y ROC-AUC como criterio secundario; el test no se usó para elegirlo.

## 6. Resultados y KPIs

**¿Cuáles son los principales resultados?**

- Recall CV: 0,8222.
- ROC-AUC CV: 0,8417.
- Recall test: 0,8091.
- Precisión test: 0,4588.
- F1 test: 0,5856.
- ROC-AUC test: aproximadamente 0,84.

**¿Cuáles son los KPIs de negocio?**

- Tasa histórica de churn: 26,54%.
- Recall de churn: identifica 301 de 372 abandonos en test.
- Captura en el 20% superior: 194 de 372 abandonos, equivalente a 52,15%.

**¿Qué significa la precisión más baja?**

Que existe un volumen relevante de falsos positivos. El modelo sirve para priorizar y debe complementarse con una revisión humana y un umbral definido según la capacidad operativa.

## 7. Uso operativo y límites

**¿Cómo se usaría el modelo?**

Se cargarían los datos vigentes, se generaría una probabilidad de churn, se ordenarían los clientes y se priorizaría la revisión de los casos de mayor riesgo. El segmento y las variables relevantes aportarían contexto antes de decidir una acción.

**¿Qué decisión apoya?**

La decisión de a qué clientes conviene priorizar para una revisión preventiva de retención. No decide automáticamente una oferta, sanción o rechazo.

**¿Qué no se puede afirmar con este proyecto?**

No se puede afirmar que una campaña evitará el churn, ni calcular retención futura, costos o impacto económico sin datos adicionales y una medición controlada.

## 8. Evidencia que cada integrante debe ubicar

Antes de la defensa, cada integrante debe poder abrir y señalar:

- [README.md](../README.md): problema, KPIs, resultados y CRISP-DM.
- [01_EDA_y_Limpieza.ipynb](../notebooks/01_EDA_y_Limpieza.ipynb): calidad de datos, EDA y preparación.
- [02_Modelado_y_Evaluacion.ipynb](../notebooks/02_Modelado_y_Evaluacion.ipynb): modelos, validación, selección, métricas y uso operativo.
- [pipeline.py](../src/pipeline.py): preprocesamiento reutilizable.
- [run_project.py](../scripts/run_project.py): flujo reproducible de partición y entrenamiento.

El subpunto se considera verificado únicamente después de que todos los integrantes realicen este ensayo o una defensa equivalente.
