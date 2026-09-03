# Proyecto de Machine Learning: Predicción de Fuga de Clientes (Telco Customer Churn)

**Metodología:** CRISP-DM  
**Tipo de Problema:** Clasificación Supervisada (Binaria)  
**Dataset:** Kaggle - Telco Customer Churn  

---

## 1. Descripción del Problema de Negocio

En la industria de las telecomunicaciones, el costo de adquisición de un nuevo cliente es significativamente más alto que el costo de mantener a uno existente. La pérdida recurrente de clientes (*Churn*) impacta directamente en los ingresos recurrentes de la organización y degrada el valor del tiempo de vida del cliente (*Customer Lifetime Value*).

### Planteamiento del Problema:
La empresa de telecomunicaciones presenta una tasa constante de cancelación de servicios sin una herramienta automatizada que permita identificar de forma anticipada qué clientes están próximos a abandonar la compañía.

### Propuesta de Solución:
Desarrollar, evaluar e implementar un modelo predictivo de Machine Learning capaz de clasificar a los clientes según su riesgo de fuga (`Churn` = `Yes`/`No`), permitiendo al equipo comercial y de retención diseñar e implementar campañas preventivas dirigidas antes de que la cancelación se efectúe.

---

## 2. Objetivos del Proyecto

- **Objetivo General:**  
  Construir y validar un modelo de clasificación supervisada para predecir la fuga de clientes utilizando información sociodemográfica, contractual y de consumo de servicios.

- **Objetivos Específicos:**
  1. Realizar un Análisis Exploratorio de Datos (EDA) exhaustivo para identificar variables e interacciones asociadas al abandono del servicio.
  2. Aplicar técnicas de limpieza y calidad de datos (tratamiento de nulos, inconsistencias de tipo y eliminación de registros duplicados).
  3. Preprocesar y transformar las variables categóricas y numéricas garantizando la prevención de fuga de información (*Data Leakage*).
  4. Entrenar, comparar y optimizar múltiples algoritmos de clasificación priorizando métricas orientadas al negocio.
  5. Documentar el flujo analítico completo bajo un entorno modular y reproducible.

---

## 3. Definición de KPIs y Métricas de Evaluación

### KPIs de Negocio:
1. **Tasa Mensual de Churn (Churn Rate):** Porcentaje de clientes que cancelan su suscripción en el periodo evaluado.
2. **Efectividad del Plan de Retención:** Proporción de clientes con alto riesgo identificados que aceptan una oferta de retención.
3. **Costo Promedio por Intervención de Retención:** Optimización del presupuesto de marketing al dirigir esfuerzos únicamente al segmento de alto riesgo.

### Métricas de Machine Learning:
- **Recall / Sensibilidad (Métrica Principal):** Maximizar el `Recall` para la clase positiva (`Churn = Yes`). Se busca minimizar los **Falsos Negativos** (clientes en riesgo de fuga no detectados por el modelo), ya que perder un cliente tiene un costo superior a enviar una oferta de retención a un cliente que no se iba a fugar.
- **Precision:** Evaluar la proporción de alertas verdaderas de retención para controlar el costo de falsos positivos.
- **ROC-AUC & PR-AUC:** Evaluar la capacidad general del modelo para separar la clase positiva de la negativa bajo escenarios de desbalance de clases.

---

## 4. Descripción de las Fuentes de Datos

El dataset utilizado corresponde al archivo público **Telco Customer Churn** disponible en Kaggle (7,043 registros y 21 atributos originales).

### Atributos Destacados:
- **Datos Demográficos:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Servicios Contratados:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Cuenta y Facturación:** `tenure` (antigüedad en meses), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Variable Objetivo (Target):** `Churn` (`Yes` = Cliente fugado, `No` = Cliente retenido).

---

## 5. Metodología Utilizada (CRISP-DM)

El proyecto se estructuró siguiendo las 6 etapas del ciclo estándar **CRISP-DM**:

1. **Comprensión del Negocio:** Definición del objetivo de retención, impacto financiero de los Falsos Negativos y alineación con métricas ML (`Recall`).
2. **Comprensión de los Datos:** Análisis de distribución del target, tipos de variables e identificación de anomalías en datos numéricos/categóricos.
3. **Preparación de los Datos:**
   - Limpieza de datos (conversión de tipos, imputación estratégica y eliminación de duplicados).
   - Codificación de variables categóricas (One-Hot Encoding).
   - Escalamiento de variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`).
   - División de conjuntos en Entrenamiento (`Train`) y Prueba (`Test`) mediante muestreo estratificado.
4. **Modelado:** Entrenamiento de modelos Baseline (Regresión Logística) y avanzados (Random Forest, XGBoost / LightGBM).
5. **Evaluación:** Diagnóstico mediante validación cruzada, matrices de confusión y curvas ROC/PR.
6. **Despliegue y Documentación:** Organización de código modular en Jupyter Notebooks y archivo `.pkl` del pipeline final.

---

## 6. Preparación y Análisis Exploratorio de Datos (EDA)

### 6.1 Calidad de Datos y Limpieza
- **Variable `TotalCharges`:** Originalmente importada como tipo texto/objeto. Se identificaron 11 registros con espacios en blanco `' '` correspondientes a clientes con `tenure = 0` meses. Se imputaron con valor `0.0` y se casteó la columna a `float64`.
- **Identificador Único:** Se eliminó la variable `customerID` por no aportar valor predictivo.
- **Registros Duplicados:** Tras eliminar el ID, se detectaron y removieron duplicados exactos para evitar sesgos de sobreajuste.

### 6.2 Hallazgos e Insights del Negocio
1. **Tipo de Contrato:** Los clientes con contrato **mes a mes (Month-to-month)** representan la mayor tasa de abandono. Los contratos a 1 o 2 años muestran un comportamiento de permanencia muy estable.
2. **Antigüedad (`tenure`):** La mayor probabilidad de cancelación ocurre en los primeros **1 a 12 meses** del cliente.
3. **Servicios de Soporte:** La presencia de servicios adicionales como `TechSupport` y `OnlineSecurity` reduce drásticamente la probabilidad de Churn.
4. **Métodos de Pago:** Los clientes que pagan con **Electronic Check** presentan una tasa de rotación significativamente superior en comparación con transferencias bancarias o tarjetas automáticas.
