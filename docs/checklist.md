# Checklist de revisión · Telco Churn + backend/frontend/docker

Esta checklist actualiza la revisión del proyecto para incluir la recomendación de negocio, la validación del modelo, la arquitectura de servicio y la entrega operativa local con Docker. La revisión integral vigente está en [revision_integral_ep2.md](./revision_integral_ep2.md).

## 00 · Estado actual del proyecto

☑ El proyecto ya define problema de negocio, KPI, dataset y flujo de ETL/modelado.

☑ La lógica de clasificación prioriza Recall y usa `class_weight='balanced'` sin SMOTE en la configuración canónica.

☑ El repositorio incluye lógica de umbral operativo en [../src/decision.py](../src/decision.py) y un artefacto persistido en [../output/modelos/telco_churn_pipeline.joblib](../output/modelos/telco_churn_pipeline.joblib).

☑ La documentación de negocio y modelo está distribuida entre [../README.md](../README.md), [../docs/guia_defensa_ep2.md](../docs/guia_defensa_ep2.md), [../docs/presentacion_ep2.md](../docs/presentacion_ep2.md) y [../docs/revision_integral_ep2.md](../docs/revision_integral_ep2.md).

☐ No existe backend API (FastAPI/Flask) ni frontend web (HTML/JS o Streamlit) ejecutable desde el repo.

☐ No existe Dockerfile, docker-compose ni despliegue local reproducible.

☐ No hay endpoint de predicción que devuelva probabilidad, porcentaje y métricas de negocio para un cliente.

☐ No está documentada la comparación explícita de opciones de backend/frontend y la decisión de arquitectura.

## 00.1 · Ruta de optimización del modelo

☑ Mantener el modelo actual como baseline y guardar una copia de respaldo para comparación.

☑ Evaluar múltiples modelos con validación cruzada priorizando Recall y PR-AUC.

☑ Incorporar Lazy Predict como comparación rápida de líneas base.

☑ Ejecutar búsqueda de hiperparámetros con Random Search y Grid Search sobre los mejores candidatos.

☑ Probar calibración del score para producir probabilidad real en escala 0-1 y 0-100.

☑ Definir umbral de decisión por negocio, no solo con 0.5.

☑ Preparar salida de API con clase binaria, riesgo score, probabilidad, porcentaje y clasificador de riesgo.

☑ Mantener evidencia gráfica y tabular del proceso para documentar cambios y decisiones en etapas posteriores.

---

## 01 · Revisión del modelo contra la recomendación de negocio

☑ El modelo actual se retiene como baseline y no se reemplaza por arquitectura nueva sin comparación.

☑ Se entiende que la discrepancia entre score y probabilidad puede ocurrir por desbalance, umbral 0.5, priorización de precisión sobre recall o falta de calibración.

☑ El score del modelo se interpreta como riesgo relativo, no como probabilidad calibrada, salvo validación adicional.

☑ La estrategia recomendada es mantener la versión base actual y ajustar el umbral de decisión sobre score, no solo cambiar el algoritmo.

☑ La versión actual del modelo quedará guardada como respaldo antes de comenzar cualquier optimización o remapeo del pipeline.

☑ Las métricas operativas priorizadas son Recall de churn, PR-AUC, F1 de churn y costo de falsos negativos frente a falsos positivos.

☑ Se quiere convertir el score en probabilidad de riesgo con salida en 0-1 y también en porcentaje 0-100 para negocio.

☑ Se reforzará la comparación con Lazy Predict y búsqueda de hiperparámetros para derivar la mejor versión posible del modelo.

☐ Faltan análisis y documentación de umbral operativo final para capacidad real y costos del negocio.

☐ Faltan validaciones de calibración de probabilidad para convertir el score en porcentaje legible por negocio.

### Optimización planificada del modelo

☐ Guardar la versión actual del pipeline actual como backup reproducible.

☐ Evaluar la calibración del score con `CalibratedClassifierCV` o regresión isotónica.

☐ Ajustar el umbral de decisión por riesgo operativo, no solo por 0.5.

☐ Generar salidas con: `predicted_churn`, `risk_score`, `probability_churn`, `probability_percent` y `label`.

☐ Mantener la lógica actual como baseline de comparación frente a la versión optimizada.

---

## 02 · Recomendación técnica: backend

### Opción recomendada: FastAPI

☑ FastAPI es la opción más apropiada para una API de predicción porque ofrece validación de entrada con Pydantic, documentación automática OpenAPI/Swagger, performance alta con ASGI y buen fit para microservicios y despliegue con Docker.

☑ Es adecuado para exponer un endpoint de predicción con probabilidades, riesgo y métricas de negocio sin mezclar lógica de presentación con lógica de inferencia.

☑ El backend separa claramente la capa de inferencia (modelo y preprocessing) de la capa de servicio (API), lo que mejora testeo, mantenimiento y escalabilidad.

### Otras opciones consideradas y descartadas

☐ Streamlit como backend: viable para prototipos rápidos, pero no es la mejor opción para una API reusable ni para un servicio de producción.

☐ Flask: más ligero y familiar, pero requiere más boilerplate manual para validación, documentación y mantenimiento de endpoints.

☐ Dash: útil para dashboards interactivos, pero no aporta ventaja clara sobre FastAPI cuando lo principal es una API de predicción con documentación y despliegue limpio.

### Justificación de decisión

La recomendación es usar FastAPI como backend porque el caso de uso es claramente un servicio de ML: input estructurado, salida con probabilidad y decisión, documentación automática, testabilidad y despliegue simple con Docker. Streamlit conviene más como capa de demo o frontend interno, no como API backend principal.

---

## 03 · Recomendación técnica: frontend

### Opción recomendada: HTML/JS + Bootstrap + FastAPI

☑ HTML + JavaScript + Bootstrap ofrece una interfaz simple, moderna y muy legible para una consulta de cliente de riesgo de churn.

☑ Bootstrap facilita una UI limpia, responsiva y más profesional sin complicar el despliegue local.

☑ Es ideal para un formulario con validación, visualización del porcentaje de riesgo y una recomendación accionable para negocio.

☑ La separación entre frontend, API y modelo es clara: la web solo consulta; la lógica de decisión y el pipeline quedan en el backend.

### Opción alternativa: Streamlit

☑ Streamlit es excelente para prototipos, validación rápida y dashboards de análisis para stakeholders no técnicos.

☑ Es muy útil para demo, exploración y validación interna, pero ofrece menos control de diseño y menos separación entre UI, negocio y backend.

### Comparativa rápida

| Opción | Ventajas | Desventajas | Cuándo conviene |
|---|---|---|---|
| HTML/JS + Bootstrap | Control visual, UI más productiva, fácil integrar con API, muy buena para web interna | Requiere más trabajo manual | Producto final o versión de negocio |
| Streamlit | Muy rápido de prototipar, demo simple | Menos flexible visualmente, menos preparada para producto serio | POC, demo interna, validación |

### Decisión sugerida

- Para despliegue local y servicio productivo: usar FastAPI como backend y HTML/JS con Bootstrap como frontend.
- Para validación interna o POC: Streamlit como opción complementaria, no como diseño final.

### Requisito del frontend

☐ Crear una página web simple con formulario para consultar un perfil de cliente.

☐ Mostrar resultado con porcentaje 0-100, nivel de riesgo y decisión binaria.

☐ Mostrar el texto: “alto riesgo”, “riesgo medio” o “bajo riesgo” según umbral operativo.

☐ Consumir la API con JavaScript y renderizar los datos en pantalla sin recargar la página.

---

## 04 · Docker local

☐ No existe Dockerfile ni docker-compose en el repositorio actual.

### Checklist de implementación sugerida

☐ Generar `Dockerfile` para la API y el modelo.

☐ Generar `docker-compose.yml` para levantar servicio local con puerto expuesto.

☐ Crear `.dockerignore` para mantener la imagen limpia.

☐ Definir `requirements.txt` o usar el entorno del proyecto con versiones fijas.

☐ Exponer el puerto local `8000` para FastAPI.

☐ Añadir una ruta de salud como `/health` para validación del contenedor.

### Comandos sugeridos

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

O bien:

```bash
docker compose up --build
```

---

## 05 · Predicción con porcentaje y métricas sugeridas

### Endpoint recomendado

☐ `POST /predict`

### Entrada sugerida

- Datos del cliente (features del modelo)
- Si se desea, un campo `customer_id` y un campo `threshold` opcional

### Salida sugerida

```json
{
  "customer_id": "CUST-001",
  "predicted_churn": true,
  "risk_score": 0.74,
  "probability_churn": 0.74,
  "probability_percent": 74.0,
  "decision_threshold": 0.5,
  "risk_label": "alto riesgo",
  "business_action": "contactar_cliente",
  "metrics_context": {
    "recall_focus": "prioridad de detección de churn",
    "precision": 0.47,
    "f1_churn": 0.60,
    "pr_auc": 0.84
  }
}
```

### Reglas recomendadas para presentación

- Exponer `probability_churn` en escala 0-1 y `probability_percent` en 0-100.
- Mantener el `risk_score` como valor bruto del modelo y la probabilidad calibrada como valor de negocio.
- Mostrar métricas sugeridas (Recall, PR-AUC, F1 de churn y costo relativo de FNs vs FPs) junto con la decisión.
- No mostrar como probabilidad “calibrada” si no hay validación de calibración.

### Versionado del modelo

☐ Guardar el modelo actual como `baseline` para regresión y comparación.

☐ Guardar la versión optimizada posterior como `v1` o `production` después de calibración y validación.

☐ Mantener el artefacto base para reintentos y benchmarks sin perder la referencia original.

---

## 06 · Checklist de cumplimiento del prompt mejorado

### Bloque 1: Configuración y contexto

☑ El proyecto tiene contexto de negocio y diccionario de variables documentado.

☐ Falta un notebook único que reúna todo el flujo completo, con Markdown y código por bloque.

### Bloque 2: Carga, inspección y split

☑ El repo sí genera train/test separados y usa holdout para evaluación final.

☑ Se aplica separación por perfil y se evita usar test hasta la evaluación final.

☐ No existe un notebook académico con “Bloque 2: test congelado” trazado explícitamente bajo la regla del prompt.

### Bloque 3: EDA y limpieza

☑ Hay exploración EDA y limpieza en notebook y scripts.

☐ No está organizado en un notebook único con bloques exactamente como pide el prompt.

### Bloque 4: Feature engineering y selección

☑ Hay selección de variables por correlación y SFS.

☑ Hay reducción final a 10 variables transformadas.

☐ Falta una estructura académica explícita del prompt, con todas las secciones y conclusiones por bloque.

### Bloque 5: Modelo no supervisado

☑ El proyecto incluye PCA/K-Means como comparación exploratoria.

☐ No está presentado como un bloque académico del notebook con conclusión y uso adecuado.

### Bloque 6: Balanceo y línea base (Lazy Predict)

☑ El proyecto usa `class_weight='balanced'`, evita SMOTE en la configuración canónica.

☐ No hay Lazy Predict implementado ni una línea base diseñada según esa instrucción.

### Bloque 7: Búsqueda de hiperparámetros

☑ Hay ajuste por Optuna y comparación de modelos con validación cruzada.

☐ No está formulado en el formato de Random Search + Grid Search + Optuna dentro de un único notebook, tal como exige el prompt.

### Bloque 8: Evaluación final con test congelado

☑ El proyecto carga test tras el ajuste final y reporta métricas.

☑ Se documenta Recall, Precision, F1 y ROC-AUC.

☐ La decisión final no está presentada con un protocolo de umbral operativo y métricas de negocio en conjunto.

### Bloque 9: Conclusiones

☑ El proyecto concluye con una narrativa defendible de negocio.

☐ Faltan conclusiones finales que integren score, umbral, costo de FNs/FPs y despliegue local para uso operativo.

---

## 07 · Puntos que fallan frente al prompt mejorado

1. No hay un único notebook que siga la estructura “Bloque 1 a Bloque 9” del prompt.
2. No se emplea Lazy Predict dentro del flujo académico ni se documenta la línea base con ese enfoque.
3. No se compara Random Search, Grid Search y Optuna con la misma métrica en el mismo notebook.
4. No existe un backend con API de predicción ni un frontend asociado.
5. No hay Docker local ni despliegue reproducible con `docker compose`.
6. No se documenta de forma explícita la justificación de FastAPI frente a Flask, Streamlit y otras alternativas.
7. No existe una interfaz que devuelva porcentaje, riesgo y métricas de negocio para el usuario final.
8. No está validado ni documentado el flujo de calibración de probabilidades antes de exponer resultados porcentuales.
9. El proyecto tiene un buen baseline analítico, pero aún no está preparado como producto de ML operable localmente.

---

## 08 · Recomendación final de cierre

El repositorio actual ya cumple bien la parte analítica y de validación de modelo para un proyecto académico/EP2: limpieza, EDA, split saneado, selección de variables y modelo canónico con Recall priorizado. Sin embargo, no cumple aún con la parte “producto listo para uso local” del prompt mejorado: backend, frontend, Docker y endpoint con porcentaje y métricas de negocio.

La recomendación práctica es:

1. Mantener el modelo actual como baseline y como versión base del producto.
2. Añadir FastAPI como backend de inferencia con endpoint `/predict`.
3. Usar HTML/JS para una UI según alcance productivo y Streamlit solo como demo o dashboard interno.
4. Preparar Docker local para ejecución reproducible.
5. Añadir la política de decisión y metricas de negocio al servicio de predicción, junto con una explicación explícita de que el score no es una probabilidad calibrada salvo validación adicional.

Con este alcance, el proyecto deja de ser solo un notebook académico y pasa a ser un flujo de ML con backend, visualización y despliegue local coherente con la decisión de negocio.
