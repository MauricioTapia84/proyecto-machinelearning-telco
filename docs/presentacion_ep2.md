# Guion de presentación EP2 · Telco Customer Churn

Duración objetivo: **9 minutos 30 segundos**. El contenido está basado en los resultados documentados en el [README](../README.md), el [notebook EDA](../notebooks/01_EDA_y_Limpieza.ipynb) y el [notebook de modelado](../notebooks/02_Modelado_y_Evaluacion.ipynb).

## Diapositiva 1 · Problema y objetivo — 1:00

- La empresa necesita priorizar clientes con riesgo de abandonar el servicio.
- El objetivo es construir un modelo que ordene clientes por probabilidad de `Churn`.
- El modelo apoya una revisión preventiva; no ejecuta campañas ni toma decisiones automáticas sobre clientes.

## Diapositiva 2 · Datos y metodología — 1:00

- Dataset Telco Customer Churn: 7.043 clientes y 19 variables predictoras originales.
- Variable objetivo binaria: `Churn`.
- Flujo CRISP-DM: comprensión, datos, preparación, modelamiento, evaluación y uso operativo.
- Se separó train/test con estratificación y se evitó utilizar el test para seleccionar el modelo.

## Diapositiva 3 · Hallazgos del EDA — 1:15

- La tasa histórica de churn es **26,54%**.
- Los contratos mes a mes muestran mayor abandono que los contratos de uno o dos años.
- Los clientes con menor antigüedad concentran más abandono.
- También se observaron diferencias por servicios de internet, método de pago y cargos mensuales.
- Son asociaciones descriptivas y no demuestran causalidad.

## Diapositiva 4 · Preparación y representaciones — 1:00

- `TotalCharges` se convirtió a numérico y se trataron sus espacios vacíos.
- Las transformaciones se ajustaron dentro del flujo de entrenamiento.
- Se compararon los predictores originales con una representación PCA + K-Means.
- Se conservaron los predictores originales porque obtuvieron mejor Recall para el objetivo de detección.

## Diapositiva 5 · Modelos comparados — 1:15

- Se evaluaron Regresión Logística, Random Forest, Gradient Boosting y SVM.
- Se utilizó SMOTE únicamente dentro del entrenamiento para tratar el desbalance.
- La selección priorizó Recall y utilizó ROC-AUC como criterio secundario en validación cruzada.
- El modelo seleccionado fue **SVM con predictores originales**.

## Diapositiva 6 · Resultados del modelo — 1:15

- Recall CV: **0,8222**.
- ROC-AUC CV: **0,8417**.
- Recall test: **0,8091**.
- Precisión test: **0,4588**; F1: **0,5856**; ROC-AUC: aproximadamente **0,84**.
- El modelo detecta una proporción alta de abandonos, pero genera falsos positivos que deben ser gestionados por el equipo.

## Diapositiva 7 · KPIs y decisión de negocio — 1:15

- KPI 1: tasa de churn histórica: **26,54%**.
- KPI 2: Recall: detecta **301 de 372** abandonos en test.
- KPI 3: el 20% de mayor riesgo concentra **194 de 372**, equivalente a **52,15%** de los abandonos observados.
- Decisión apoyada: definir a qué clientes priorizar para una revisión preventiva según la capacidad disponible.

## Diapositiva 8 · Recomendaciones y límites — 1:15

- Ordenar la cartera por probabilidad de churn y comenzar por los casos de mayor riesgo.
- Revisar el contexto del cliente y su segmento antes de definir una acción.
- Mantener supervisión humana y ajustar el umbral según capacidad operativa real.
- Medir posteriormente el resultado de las campañas con datos observados y, si es posible, un grupo de comparación.
- El modelo no estima por sí solo retención futura, costos ni efecto causal de una campaña.

## Cierre — 0:15

El proyecto entrega una herramienta de priorización basada en riesgo: permite concentrar la revisión comercial en una parte acotada de la cartera, manteniendo explícitos sus métricas, límites y necesidad de validación posterior.
