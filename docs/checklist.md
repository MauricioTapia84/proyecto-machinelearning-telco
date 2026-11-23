# Checklist de revisión · Telco Churn + backend/frontend/docker

Esta checklist actualiza la revisión del proyecto para incluir la recomendación de negocio, la validación del modelo, la arquitectura de servicio y la entrega operativa local con Docker. La revisión integral vigente está en [revision_integral_ep2.md](./revision_integral_ep2.md).

## Estado verificado del repositorio

Esta sección reemplaza el estado inicial desactualizado de la checklist. Las marcas siguientes se basan en el código y artefactos vigentes del repositorio.

☑ El proyecto define problema de negocio, KPI, dataset y flujo ETL/modelado.

☑ La clasificación prioriza Recall y usa `class_weight='balanced'` sin SMOTE en el flujo canónico.

☑ Existe FastAPI con `GET /health`, `POST /predict` y una interfaz HTML/JS integrada en `app/main.py`.

☑ Existe `Dockerfile`, `.dockerignore` y `docker-compose.yml` para ejecución local.

☑ Existe lógica de umbral económico en `src/decision.py` y el artefacto actual guarda umbral y política operativa.

☑ `/predict` devuelve clase, score, aproximación porcentual, nivel de riesgo, acción y contexto de métricas.

☑ El flujo de inferencia acepta perfiles raw y matrices procesadas mediante `src/inference.py`.

☑ Falta ejecutar y documentar una validación Docker real en un host con Docker activo.

☑ El API ya identifica explícitamente si su salida porcentual es una aproximación sigmoid o una probabilidad calibrada.

☐ Falta calibrar formalmente el score para presentar una probabilidad estadísticamente válida: la aproximación actual no se llama calibrada.

☐ Falta sincronizar totalmente el artefacto generado por el notebook con el artefacto generado por `src/train.py`.

☐ Falta actualizar varias afirmaciones históricas de README, checklist y notebook.

## Pendientes consolidados

### Prioridad crítica

1. **Proteger el artefacto canónico**
   - Evitar que una celda del notebook sobrescriba `telco_churn_pipeline.joblib` con `operating_threshold: None`.
   - El notebook debe guardar el mismo contrato de metadata que `src/train.py`.
   - Verificar que `operating_threshold`, `operating_policy`, `score_method` y `feature_columns` sean consistentes.

2. **Validar Docker de extremo a extremo**
   - Ejecutar `docker compose config`.
   - Ejecutar `docker compose up --build` en una máquina con Docker activo.
   - Validar `GET /health` y `POST /predict` dentro del contenedor.
   - Documentar únicamente como probado lo que haya sido ejecutado realmente.

3. **Sincronizar documentación e implementación**
   - Actualizar [README.md](../README.md) para indicar que la API acepta perfiles raw mediante el preprocesamiento de inferencia.
   - Corregir nombres vigentes y métricas históricas en README, CHANGELOG y documentos técnicos.
   - Reemplazar afirmaciones obsoletas de la checklist: no declarar como faltantes componentes ya implementados.

### Prioridad alta

4. **Formalizar la salida probabilística**
   - Evaluar `CalibratedClassifierCV` con `sigmoid` y/o regresión isotónica mediante CV de entrenamiento.
   - Conservar el `decision_function` original si solo sirve como score de riesgo relativo.
   - No renombrar `_sigmoid` como probabilidad calibrada.
   - Si se obtiene calibración válida, persistir el método y validar Brier score/calibration curve.
   - Evaluar si `probability_churn` debe representar probabilidad calibrada o aproximación etiquetada explícitamente.

5. **Completar metadata de métricas del artefacto**
   - Persistir Precision, Recall, F1, ROC-AUC, PR-AUC y umbral final de test.
   - Hacer que `app/main.py` lea métricas reales del artefacto sin valores por defecto ambiguos.
   - Añadir pruebas que impidan que el contrato devuelva métricas inventadas o cero silenciosamente.

6. **Unificar el criterio operativo en todas las capas**
   - Usar el mismo umbral persistido en `score_customers()`, `/predict`, scripts de negocio y métricas por grupo.
   - Corregir scripts que todavía reportan el umbral 0.0 o `pipeline.predict()` si el objetivo es evaluar la política guardada.
   - Documentar claramente qué métricas usan el threshold por defecto y cuáles usan el threshold operativo.

7. **Actualizar y depurar el notebook de modelado**
   - Mantener una sola celda Lazy Predict o utilizar una implementación única importada de forma defensiva.
   - Eliminar duplicados de importaciones y celdas redundantes.
   - Actualizar todas las cifras impresas para que correspondan al artefacto y ejecutaciones vigentes.
   - Sustituir afirmaciones futuras o pendientes por conclusiones verificadas.

### Prioridad académica

8. **Organizar el notebook en bloques académicos 1–9**
   - Bloque 1: configuración y carga.
   - Bloque 2: split y test congelado.
   - Bloque 3: EDA y limpieza.
   - Bloque 4: feature engineering y selección.
   - Bloque 5: PCA/K-Means con límites explícitos.
   - Bloque 6: Lazy Predict y comparación de línea base.
   - Bloque 7: búsqueda de hiperparámetros Random Search, Grid Search y Optuna.
   - Bloque 8: evaluación final y umbral operativo.
   - Bloque 9: conclusiones, límites y despliegue.

9. **Estandarizar Lazy Predict**
   - Ejecutar Lazy Predict solo dentro de una celda opcional, sin romper el notebook si falta el paquete.
   - Guardar la comparación en `output/resultados/lazy_predict_recall_cv.csv` con nombre único.
   - Documentar versiones de scikit-learn y lazypredict usadas.
   - No permitir que Lazy Predict consulte test ni altere la selección final.

10. **Documentar el protocolo de búsqueda**
    - Comparar Random Search, Grid Search y Optuna con métricas y folds idénticos.
    - Mantener test fuera de todas las búsquedas.
    - Registrar semilla, número de trials, timeout, mejor score y parámetros.
    - Explicar por qué el SVM con `C≈0.1683` y `class_weight='balanced'` gana o permanece como baseline.

11. **Cerrar el bloque académico de resultados**
    - Integrar Recall, Precision, F1, ROC-AUC, PR-AUC, capacidad y costo.
    - Presentar matriz de confusión y escenarios de umbral.
    - Señalar explícitamente que el test fue explorado por el analista y que se requiere una cohorte temporal o externa.

### Prioridad de producto y documentación

12. **Documentar decisiones de arquitectura**
    - Agregar una tabla FastAPI vs Flask vs Streamlit/Dash con ventajas y desventajas.
    - Justificar FastAPI y HTML/JS como diseño elegido.
    - Documentar cómo se separan backend, inferencia y frontend.

13. **Completar validaciones HTTP reales**
    - Mantener pruebas automatizadas para `/health` y `/predict`.
    - Añadir una prueba manual o automatizada que arranque Uvicorn y haga una petición HTTP real.
    - Añadir validación de campos obligatorios y respuestas de error 400/503.

14. **Completar el cierre operativo**
    - Documentar el arranque local con Conda y el arranque con Docker.
    - Documentar variables de entorno numéricas de Windows.
    - Explicar que el servicio usa un score aproximado hasta finalizar calibración.
    - Agregar instrucciones de monitorización y reentrenamiento futuro.

15. **Actualizar el nombre del notebook vigente**
    - Decidir si el archivo académico será `02_Modelado_y_Evaluacion.ipynb` o el nombre ejecutado será el único activo.
    - Evitar que README, docs y AGENTS se refieran a un archivo que no existe.
    - Mantener evidencia ejecutada sin crear notebooks duplicados o históricos.

16. **Revisar artefactos históricos y limpieza continua**
    - Verificar archivos `_legacy`, `_backup`, `_old`, checkpoints y cachés antes de cerrar el proyecto.
    - Conservar solo `telco_churn_pipeline.joblib` como artefacto canónico salvo que exista una comparación versionada explícita.
    - Evitar que el notebook regenere copias con nombres rivales o tablas obsoletas.

---

## 07 · Puntos que fallan frente al prompt mejorado

1. [ ] El notebook sigue sin estar organizado como una estructura única “Bloque 1 a Bloque 9”.
2. [ ] Lazy Predict puede duplicarse o faltar en la ejecución académica.
3. [ ] Falta una comparación reproducible única de Random Search, Grid Search y Optuna.
4. [ ] La documentación Docker debe contar una ejecución real, no solo archivos preparados.
5. [ ] La justificación de arquitectura debe quedar consolidada fuera de comentarios dispersos.
6. [ ] Falta calibración probabilística formal antes de presentar `probability_churn` como probabilidad real.
7. [ ] Falta sincronizar README, checklist, notebook y artefacto con el estado vigente.
8. [ ] Falta integrar coste de FN/FP, umbral operativo y métricas finales en una conclusión única.

---

## 08 · Recomendación final de cierre

El repositorio ya cuenta con modelo, pipeline de inferencia, API, frontend, Docker y una suite de pruebas. El cierre debe priorizar consistencia y validación real:

1. Hacer que el notebook genere el mismo artefacto canónico que el script oficial.
2. Calibrar o etiquetar explícitamente la salida probabilística.
3. Sincronizar artefacto, API, scripts, notebooks y documentación.
4. Validar Docker en un host real.
5. Convertir el notebook activo en una entrega académica única y reproducible.
6. Mantener test reservado y no volver a usarlo para seleccionar modelo, variables o umbral.

---

## Avance actualizado — 23 de noviembre de 2026

### Completado

☑ **Artefacto canónico protegido desde el notebook de modelado**
   - La celda final ya deriva el umbral desde scores out-of-fold de train, no desde test.
   - La metadata incluye `operating_threshold`, `operating_policy`, `feature_columns` y métricas test.
   - La suite vigente sigue pasando: 17 pruebas.

☑ **Crash de SequentialFeatureSelector aislado y corregido**
   - El default de SFS pasó de `LinearRegression` a `LogisticRegression` balanceada.
   - El cambio evita el crash native observado en Windows con `LinearRegression`.
   - `LinearRegression` permanece disponible solo como opción explícita de compatibilidad.
   - Validado con 5.617 filas y 46 columnas; se seleccionaron 10 predictores sin reiniciar el proceso.

☑ **Lazy Predict duplicado eliminado del notebook activo**
   - Se retiró la segunda celda redundante que ejecutaba la misma comparación.
   - La primera celda conserva importación defensiva y salida opcional.

☑ **Documentación principal sincronizada**
   - [README.md](../README.md) ya describe LogisticRegression balanceada como screening estable.
   - Ya indica que la inferencia acepta perfiles raw y matrices procesadas.
   - [docs/guia_defensa_ep2.md](guia_defensa_ep2.md) usa la misma definición del estimador de screening.
   - La suite documentada pasó de 13 a 17 pruebas verificadas.

☑ **Docker runtime validado**
   - `Dockerfile`, `.dockerignore` y `docker-compose.yml` existen y están configurados.
   - Docker Desktop está activo y `docker compose up --build -d` fue ejecutado.
   - El contenedor `telco-churn-api` quedó en estado `healthy`.
   - Se eliminó `output/` de `.dockerignore` para incluir el artefacto canónico.
   - Se retiró el atributo obsoleto `version` de `docker-compose.yml`.

☑ **Direcciones HTTP verificadas**
   - `/` responde `200` y muestra el formulario integrado.
   - `/health` responde `200` con estado `ok` y metadata del artefacto.
   - `/predict` responde `200` con score, decisión, aproximación porcentual y métricas reales.
   - `/docs` responde `200`.
   - `/openapi.json` responde `200`.
   - `/favicon.ico` responde `200`.

☑ **Dashboard de negocio implementado**
   - La raíz presenta una interfaz tipo dashboard con sidebar, formulario y panel de resultados.
   - El campo `customer_id` fue retirado del formulario y del contrato `/predict`.
   - Los resultados incluyen gauge porcentual, score, umbral, decisión y barras de métricas.
   - El formulario consume `/predict` sin recargar la página y muestra errores inline.
   - El dashboard HTML/JS es responsive, usa perfiles sin identificador y explica resultados para públicos administrativo y técnico.

### Pendiente para cerrar esta tarea

☐ Ejecutar desde cero las celdas del notebook activo y confirmar que el artefacto conserva el mismo contrato que `src/train.py`.

## Siguiente tarea prioritaria

La siguiente tarea es ejecutar el notebook completo desde cero con el entorno `telco-ml` y confirmar que su celda final guarda exactamente el contrato del artefacto canónico: `operating_threshold`, `operating_policy`, `probability_method`, métricas de test y `feature_columns`.

Después se debe abordar formalmente la calibración probabilística. Hasta entonces, `probability_churn` permanece como aproximación sigmoid y no debe presentarse como probabilidad calibrada.
