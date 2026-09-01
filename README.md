# Predicción de Abandono de Clientes (Churn) en una Empresa de Telecomunicaciones

## 1. Descripción del Problema de Negocio

**Contexto:** Una empresa de telecomunicaciones busca reducir la tasa de abandono de sus clientes (churn). El churn es un problema crítico, ya que retener a un cliente existente es significativamente más barato que adquirir uno nuevo.

**Problema:** La empresa necesita identificar de manera proactiva a los clientes con alta probabilidad de darse de baja en el corto plazo. El objetivo es implementar estrategias de retención focalizadas (ej. ofertas personalizadas, mejoras en el servicio) para estos clientes, optimizando así la inversión en programas de fidelización.

## 2. Objetivos del Proyecto

**Objetivo General:**
Desarrollar un modelo de Machine Learning que prediga la probabilidad de que un cliente abandone la empresa, permitiendo la toma de decisiones proactivas para su retención.

**Objetivos Específicos:**
1.  Realizar un análisis exploratorio de datos (EDA) exhaustivo para comprender los patrones y factores asociados al churn.
2.  Limpiar y preparar los datos para el modelado, manejando valores nulos y codificando variables categóricas.
3.  Entrenar y evaluar múltiples modelos de clasificación (ej. Regresión Logística, Random Forest, XGBoost) para predecir el churn.
4.  Seleccionar el mejor modelo basado en métricas de rendimiento relevantes para el negocio (ej. Recall, Precisión, F1-Score, AUC-ROC).
5.  Identificar las características más influyentes en la predicción del churn para generar insights de negocio.

## 3. Definición de KPIs para el Problema de Negocio

Los KPIs deben medir el éxito del proyecto desde una perspectiva de negocio:

| KPI | Descripción | Fórmula / Métrica |
| :--- | :--- | :--- |
| **Tasa de Churn Reducida** | Reducción porcentual en la tasa de abandono mensual. | `(Churn_actual - Churn_nuevo) / Churn_actual * 100` |
| **Efectividad de la Retención** | Porcentaje de clientes identificados como "en riesgo" que aceptan una oferta de retención y permanecen. | `(Clientes_retenidos / Clientes_en_riesgo_contactados) * 100` |
| **Valor de Vida del Cliente (CLV) Protegido** | Medir el CLV promedio de los clientes retenidos gracias a la intervención. | `CLV_promedio_retenidos` |
| **Recall (Sensibilidad)** | Capacidad del modelo para identificar correctamente a los clientes que realmente abandonan. | `VP / (VP + FN)` (Muy importante para no perder clientes). |
| **AUC-ROC** | Capacidad del modelo para distinguir entre clientes que abandonan y los que no. | Área bajo la curva ROC. |

## 4. Descripción de las Fuentes de Datos

- **Fuente:** Dataset "Telco Customer Churn" de IBM, disponible en Kaggle.
- **Enlace:** [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Descripción:** El dataset contiene información de 7043 clientes y 21 características (features) que incluyen:
    - **Datos Demográficos:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
    - **Datos de Servicios Contratados:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
    - **Datos de Cuenta:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
    - **Variable Objetivo:** `Churn` (Yes/No), que indica si el cliente se fue en el último mes.

## 5. Preparación y Análisis Exploratorio de los Datos (EDA)

> **Aquí debes incluir un resumen de tu notebook.** Describe los pasos clave y muestra gráficos relevantes (puedes enlazar a la carpeta `images/`).

- **Análisis de Calidad de Datos:**
    - Se identificó que `TotalCharges` es un tipo de dato `object` que debe convertirse a `float`.
    - Se encontraron 11 valores nulos en la columna `TotalCharges`, los cuales fueron eliminados al representar menos del 1% del total.
    - No se encontraron valores atípicos extremos en las variables numéricas (`tenure`, `MonthlyCharges`).

- **Análisis Univariado y Bivariado:**
    - **Distribución de Churn:** Aproximadamente el 26.5% de los clientes abandonaron el servicio, lo que indica un desbalanceo en la variable objetivo (esto es importante para el modelado).
    - **Tenencia (Tenure):** Los clientes con menor antigüedad (`tenure` bajo) tienden a tener una tasa de churn mucho más alta. `[Incluir gráfico: images/churn_vs_tenure.png]`
    - **Contrato:** Los clientes con contratos mensuales (mes a mes) tienen una probabilidad de churn significativamente mayor que aquellos con contratos anuales o de dos años.
    - **Servicios:** La falta de servicios adicionales como `OnlineSecurity` o `TechSupport` está fuertemente relacionada con el abandono.

- **Preparación y Transformación:**
    - **Codificación:** Las variables categóricas nominales (ej. `gender`, `InternetService`) se codificaron usando One-Hot Encoding. Las variables ordinales binarias (ej. `Partner`, `Churn`) se mapearon a valores numéricos (0/1).
    - **Escalado:** Las variables numéricas (`MonthlyCharges`, `tenure`) se estandarizaron (escalaron) para mejorar el rendimiento de algunos modelos (ej. Regresión Logística, SVM).
    - **Balanceo:** Se consideró el uso de técnicas de sobremuestreo (ej. SMOTE) para tratar el desbalanceo de la clase `Churn`.

## 6. Metodología Utilizada (CRISP-DM)

El proyecto se guió por la metodología **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, siguiendo sus fases:

1.  **Comprensión del Negocio:** Definición del problema, objetivos y KPIs.
2.  **Comprensión de los Datos:** Exploración inicial, identificación de la fuente y las variables. [Enlace al Notebook 01]
3.  **Preparación de los Datos:** Limpieza, transformación y feature engineering.
4.  **Modelado:** Entrenamiento de diferentes algoritmos (Regresión Logística, Random Forest, etc.) usando validación cruzada para ajustar hiperparámetros. [Enlace al Notebook 02]
5.  **Evaluación:** Comparación del rendimiento de los modelos con métricas como Recall, Precisión y AUC-ROC, seleccionando el mejor para el problema de negocio.
6.  **Despliegue (Conceptual):** El modelo final se guarda para su potencial integración en un sistema que etiquete a los clientes "en riesgo".

**Próximos Pasos (Entregables):**
- [x] Informe técnico (este archivo).
- [x] Notebooks con el desarrollo completo (`/notebooks/`).
- [x] Datos y estructura del proyecto.
- [ ] Despliegue de un prototipo (si aplica, fuera del alcance de esta evaluación).