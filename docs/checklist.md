# Checklist real del proyecto · Telco Churn

Esta checklist refleja el estado verificable del repositorio al 23 de noviembre de 2026. La revisión histórica se conserva solo como referencia; las marcas siguientes prevalecen sobre afirmaciones antiguas.

## Estado implementado y verificado

☑ Problema de negocio, KPI, ETL, EDA y split train/test definidos.

☑ Flujo canónico sin SMOTE: `class_weight='balanced'` en el SVM lineal.

☑ Selección por correlación absoluta mayor que 0.95 y SFS con 10 variables.

☑ Entrenamiento y Optuna utilizan exclusivamente train; test se carga para evaluación final.

☑ FastAPI implementada con `GET /health`, `POST /predict`, Swagger y OpenAPI.

☑ Dashboard HTML/JS integrado, responsive, narrativo y en español.

☑ Frontend sin `customer_id`, con perfiles seleccionables que autocompletan y evalúan.

☑ Dashboard con gauge, score, umbral, recomendación, matriz de confusión y barras por permanencia.

☑ Docker Desktop, `Dockerfile`, `.dockerignore` y `docker-compose.yml` validados localmente.

☑ Contenedor `telco-churn-api` ejecutándose como `healthy` en el puerto 8000.

☑ `/`, `/health`, `/predict`, `/docs`, `/openapi.json` y `/favicon.ico` responden correctamente.

☑ Inferencia raw y procesada implementada en `src/inference.py`.

☑ Artefacto canónico: `output/modelos/telco_churn_pipeline.joblib`.

☑ Artefacto isotónico opcional: `output/modelos/telco_churn_pipeline_isotonic.joblib`.

☑ Suite vigente: 28 pruebas automatizadas pasan.

☑ Notebook único: se conserva `02_Modelado_y_Evaluacion_ejecutado.ipynb` sin sufijo duplicado y fue reemplazado por la ejecución vigente.

☑ Alertas de deriva: `scripts/evaluate_retraining_alert.py` exige periodos consecutivos y nunca reentrena automáticamente.

☑ CI Docker: workflow levanta el contenedor, ejecuta smoke HTTP y carga ligera.

☑ Backups reales: `scripts/backup_model_artifact.py` y rollback SHA-256 verificable implementados y probados.

☑ Mitigación por grupo: criterio formalizado en `docs/criterio_mitigacion_grupos.md`.

☑ Notebook completo ejecutado con `telco-ml` mediante nbconvert.

☑ Autenticación opcional con `TELCO_API_KEY` implementada en `/predict`.

☑ Logs HTTP estructurados JSON implementados sin request body ni datos de clientes.

☑ CI creada en `.github/workflows/ci.yml` para pytest, deriva y Docker Compose.

☑ Equidad disponible con intervalos Wilson 95% y advertencias de soporte mínimo.

☑ Protocolos de piloto, reentrenamiento y rollback documentados en `docs/operacion_modelo.md`.

☑ Artefacto versionado v2 con esquema `telco-churn-pipeline-v2`.

☑ Hash SHA-256 persistido en `output/modelos/telco_churn_pipeline.sha256.txt` y expuesto por `/health` y `/predict`.

☑ Reporte PSI de deriva generado en `output/resultados/monitorizacion_drift.csv`.

☑ Métricas por grupo y análisis de permanencia generados en `output/resultados/metricas_por_grupo_test.csv`.

☑ Umbral OOF candidato persistido como política provisional, no aprobada.

☑ Probabilidad marcada explícitamente como aproximación sigmoid o calibración isotónica.

☑ Se verificaron pruebas HTTP reales sobre Uvicorn y Docker.
   - Uvicorn local respondió `200` en `/health` y `/predict`.
   - Docker respondió `200` en `/health` y `/predict`.
   - [scripts/smoke_test_api.py](../scripts/smoke_test_api.py) valida el contrato de salida mínimo.

☑ **Notebook activo organizado y sincronizado en secciones académicas**
   - Lazy Predict existe como una sola comparación opcional.
   - Las métricas vigentes se mantienen en conclusiones y cierre.
   - Falta una ejecución completa desde cero para regenerar todos los outputs almacenados.

☑ **Estado de política formalizado en API y artefacto**
   - El threshold actual se identifica como provisional y no aprobado.
   - El API expone `operating_policy_status='provisional'` y validación externa pendiente.
   - El dashboard advierte que no representa una decisión comercial automática.

## Resultados vigentes

### Modelo seleccionado

- Modelo: SVM lineal balanceado ajustado por Optuna.
- Variables seleccionadas: 10.
- Test: 1.409 clientes.
- Abandonos reales: 374.
- Permanencias reales: 1.035.
- TP: 310.
- FP: 302.
- FN: 64.
- TN: 733.
- Accuracy: 74.02%.
- Precision: 50.65%.
- Recall: 82.89%.
- F1: 62.88%.
- ROC-AUC: 85.13%.
- PR-AUC: 68.54%.
- Diagnóstico en test: 362 seleccionados; 229 TP, 133 FP; validación externa pendiente.

Nota: la matriz anterior corresponde a la última ejecución canónica verificable. Las ejecuciones históricas deben conservarse como evidencia, no mezclarse con el artefacto vigente.

### Umbral provisional

- Método: cuantil 75 de scores OOF.
- Threshold vigente: `0.7245316291913046`.
- Estado: recalibrado OOF; validación externa pendiente.
- No está aprobado como política comercial.
- Debe validarse nuevamente antes de cambiar contactos, capacidad o costos.

### Calibración

Comparación entrenamiento con 5 folds:

| Método | Brier | ROC-AUC |
|---|---:|---:|
| Isotónico | 0.136096 | 0.841004 |
| Sigmoid | 0.136292 | 0.841652 |

El pipeline canónico conserva el score SVM y usa una aproximación sigmoid. La calibración isotónica existe como artefacto separado, no reemplaza el baseline hasta validar estabilidad y tradeoff Recall/Precision fuera del test.

## Pendientes reales del proyecto

### Prioridad crítica

1. **Cerrar la sincronización notebook/artefacto**
   - Ejecutar el notebook completo desde cero con `telco-ml`.
   - Confirmar que su SFS, Optuna, test y umbral coincidan exactamente con `src/train.py` y el artefacto.
   - Documentar el `C` vigente y retirar cifras históricas contradictorias.

2. **Validar el umbral operativo externamente**
   - Probar la política con una cohorte temporal o externa.
   - Aprobar Recall, precision, falsos positivos y capacidad antes de convertir el candidato en política definitiva.
   - Medir costos reales de contacto, pérdida por churn y eficacia de retención.

3. **Decisión formal sobre probabilidades: mantener sigmoid y advertir explícitamente**
   - La salida canónica conserva `probability_method='sigmoid_approximation'`.
   - `probability_calibrated=False` no admite presentar el porcentaje como calibrado.
   - El artefacto isotónico es opcional y no reemplaza el baseline sin validación externa.

4. **Formalizar el contrato de producción**
   - ☑ Artefacto v2, fecha, esquema y SHA-256 disponibles.
   - Añadir autenticación si el servicio deja localhost.
   - Registrar logs estructurados de solicitud, versión de artefacto y resultado sin datos sensibles.
   - Añadir políticas explícitas de errores, timeouts y alertas.

5. **Completar automatización de calidad**
   - No existe CI configurada en `.github/workflows/`.
   - Añadir `pytest`, ejecución mínima canónica y validación Docker Compose.
   - Añadir linting/formato y carga HTTP opcional.

### Prioridad académica

6. **Organizar el notebook académico**
   - Bloque 1: configuración y contexto.
   - Bloque 2: carga, split y test congelado.
   - Bloque 3: EDA y limpieza.
   - Bloque 4: feature engineering y SFS.
   - Bloque 5: PCA/K-Means y límites.
   - Bloque 6: Lazy Predict y línea base.
   - Bloque 7: Random Search, Grid Search y Optuna.
   - Bloque 8: test final y umbral.
   - Bloque 9: conclusiones, límites y despliegue.

7. **Limpiar el notebook activo**
   - Mantener una sola celda Lazy Predict opcional.
   - Eliminar importaciones y secciones redundantes.
   - Reemplazar cifras impresas históricas por cifras del último artefacto vigente.
   - Corregir textos y caracteres dañados por codificación.

8. **Alinear documentación académica**
   - Unificar nombres de notebook y métricas en README, CHANGELOG, guía y presentación.
   - Sustituir referencias históricas a `LinearRegression` como screening canónico.
   - Documentar cuantil 75 y validación externa pendiente del umbral.

### Prioridad operativa

9. **Añadir versionado y monitorización del modelo**
   - Mantener hash o versión del artefacto.
   - Registrar deriva de variables y desempeño por periodo.
   - Definir umbral de alerta y protocolo de reentrenamiento.

10. **Validar equidad con soporte suficiente**
    - Revisar grupos con tamaño y soporte suficientes.
    - Presentar intervalos sin afirmar causalidad o injusticia automática.
    - Definir criterios para mitigación supervisada.

11. **Medir impacto real**
    - Ejecutar piloto controlado.
    - Medir tasa real de retención y costo neto.
    - No presentar métricas históricas como evidencia causal de retención.

12. **Completar operación local reproducible**
    - Documentar arranque Conda y Docker.
    - Agregar pruebas reales de `/health` y `/predict`.
    - Documentar backup, rollback y reinicio.

## Trabajo no bloqueante pero recomendable

- Añadir CI para `pytest`, validación de Docker Compose y ejecución mínima del script canónico.
- Agregar formato y linting de Python.
- Automatizar limpieza de cachés y artefactos históricos.
- Agregar una pequeña pantalla de estado del artefacto en el dashboard.
- Añadir pruebas de carga para el contenedor local.

## Criterio de cierre

El proyecto podrá considerarse terminado operacionalmente cuando:

1. El notebook completo reproduzca el artefacto canónico.
2. El umbral tenga aprobación externa y datos comerciales reales.
3. La salida probabilística esté calibrada o etiquetada como aproximada sin ambigüedad.
4. La API, dashboard y Docker estén respaldados por pruebas reproducibles.
5. README, checklist, notebook y presentación tengan métricas y nombres consistentes.

## Validación externa pendiente

☐ No existe en `data/` una cohorte temporal o externa adicional.

- El holdout actual es 20% interno del dataset disponible y ya fue usado para diagnóstico.
- No debe reutilizarse como validación externa independiente.
- Falta una cohorte nueva o temporal no explorada antes de aprobar el umbral.
- Falta un piloto controlado para medir retención real y costo operativo.

## Decisión pendiente sobre calibración

☐ Elegir si el servicio conserva la aproximación sigmoid o adopta isotónica.

- Sigmoid: Brier 0.136292 y ROC-AUC 0.841652 en CV train.
- Isotónica: Brier 0.136096 y ROC-AUC 0.841004 en CV train.
- Isotónica existe como artefacto separado.
- No reemplazar el baseline sin validación temporal externa y comparación del tradeoff Recall/Precision.

## Estado restante resumido

Completado:
- API, dashboard, Docker y pruebas HTTP locales.
- 21 pruebas automatizadas.
- Artefacto v2, SHA-256, fecha y exposición en health/predict.
- Reporte PSI de deriva sin alertas actuales.
- Documentación académica alineada con métricas vigentes.

Pendiente crítico:
1. Cohorte temporal o externa y aprobación del umbral.
2. Piloto controlado que mida retención real y costo neto.
3. Autenticación y logs estructurados si se publica fuera de localhost.
4. Ejecución completa del notebook desde el kernel `telco-ml` y limpieza final de outputs.

Pendiente operativo/recomendado:
- CI para pruebas y Docker Compose.
- Formato, linting y pruebas de carga.
- Intervalos de confianza y soporte suficiente para equidad.
- Dashboard de estado temporal y protocolización de reentrenamiento.

## Validación externa bloqueada por disponibilidad de datos

☐ **Cohorte temporal o externa:** no existe una cohorte adicional en el repositorio.

☐ **Piloto controlado:** no existe permiso comercial ni datos de intervención/retención observada.

- Estas tareas no pueden completarse localmente con el holdout actual.
- Suministrar una cohorte nueva o temporal y el diseño piloto para continuar.

## Cobertura de rúbrica EP3

☑ **IE9 Optimización de hiperparámetros (20%)**: Optuna ajusta `C` del SVM sobre train con tres folds y el modelo final se valida con cinco folds.

☑ **IE10 Técnicas de ensamble (30%)**: Voting soft y Stacking están implementados y comparados por CV y test reservado.

☑ **IE11 Generalización (20%)**: validación cruzada estratificada, test reservado y métricas Recall/Precision/F1/ROC-AUC/PR-AUC.

☑ **IE12 Solución analítica óptima (30%)**: la selección conserva el SVM ajustado por prioridad Recall, contrastando el ensamble y justificando el tradeoff.
