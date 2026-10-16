# Revisión integral del proyecto EP2

**Base de la revisión:** commit `14d4b76` (`3.5. entrega y defensa refinadas`), con avances de corrección posteriores en el árbol de trabajo.

**Alcance:** README, checklist, changelog, documentación de defensa, notebooks, scripts, código reutilizable, datos procesados, rutas y artefactos presentes.

## Dictamen ejecutivo

La documentación de negocio está más completa: explica el problema, KPIs, CRISP-DM, modelos, hallazgos de EDA y límites de uso. El notebook de modelado se reestructuró para corregir los problemas P0 y se ejecutó de principio a fin en conda; las métricas de esa ejecución ya se documentaron. El proyecto aún no está completamente cerrado: faltan sensibilidad de duplicados, unificación del flujo y artefacto de inferencia, y una política operativa para umbrales/capacidad.

**Prioridad restante antes de defender los resultados:** justificar la política de duplicados mediante análisis de sensibilidad; unificar entrenamiento, rutas y persistencia; probar inferencia con datos nuevos; y completar los supuestos operativos de contacto.

La revisión inicial fue estática porque todavía no había un entorno Python/conda disponible. Desde entonces se instaló Miniforge y se creó el entorno `telco-ml` (Python 3.11.17). Cuatro pruebas de preparación/pipeline pasan; `compileall`, validación de sintaxis y `git diff --check` también. El usuario ejecutó el EDA; la nueva versión del notebook 02 se ejecutó desde un kernel limpio de conda sin errores en sus ocho celdas de código, incluidas las cinco pruebas de Optuna. El notebook ejecutado y sus salidas se guardaron. La selección CV final fue SVM ajustado (Recall 0,8007 ± 0,0388; ROC-AUC 0,8471 ± 0,0186); el test reservado obtuvo Recall 0,7715 y ROC-AUC 0,8405. El README se actualizó con las métricas verificadas. Esto valida el protocolo del notebook, pero no unifica todavía los otros scripts/artefactos de entrenamiento.

## Hallazgos que deben corregirse

### P0 — Validez de la evaluación y resultados

1. **Corregido y verificado en ejecución limpia.** La nueva versión del notebook 02 parte de los registros sin transformar y usa `build_evaluation_pipeline`: limpieza determinista, imputación, escalado, codificación, PCA/K-Means (en la rama que lo evalúa) y SMOTE forman parte del pipeline que `cross_validate` ajusta en cada fold. La ejecución completa terminó sin errores.

   La CV de cinco folds registró medias y desviaciones; el SVM base y la variante PCA/K-Means se compararon sin consultar el test.

2. **Corregido y verificado.** La comparación, elección de representación y búsqueda Optuna usan solo `X_train`/`y_train`. El test se predice después de cerrar modelo y parámetros, en una única sección de reporte final. No debe elegirse otro candidato ni ajustarse un umbral tras inspeccionar esas métricas.

   Optuna completó cinco pruebas en CV de entrenamiento, optimizando Recall, con semilla fija. El candidato Optuna se volvió a puntuar con la misma CV de cinco folds de los modelos base.

3. **Corregido y verificado.** Se guardaron las salidas de la nueva ejecución limpia y se actualizaron las cifras principales del README: Recall test 0,7715; ROC-AUC test 0,8405; captura en el 20% superior 192/372 (51,61%). La matriz tiene 759 TN, 274 FP, 85 FN y 287 TP.

   **Pendiente:** revisar otras salidas históricas del EDA y mantener sincronizados README y cualquier presentación cuando se incorporen resultados nuevos.

4. **Corregido y verificado.** La importancia por permutación se calcula en una partición explicativa interna de `X_train`, separada de los datos con los que se ajusta ese pipeline explicativo. No usa `X_test` ni `y_test`.

   La figura y la tabla se generaron en la ejecución; la permutación usa predictores originales y no establece causalidad.

### P1 — Datos, flujo reproducible e inferencia

5. **La regla acordada elimina perfiles iguales aunque tengan IDs distintos; esto no confirma identidad.** El usuario eligió remover perfiles idénticos (incluyendo `Churn`) aunque cambie `customerID`. Se agregó `src.data.deduplicate_customer_profiles` para normalizar valores vacíos y cargos de clientes con antigüedad cero antes de deduplicar; notebook 01, `src/split_data.py`, `src/pipeline.py` y el script de KPI ahora usan esa política. La comprobación con conda encontró 22 eliminaciones, 7.021 filas retenidas y cero perfiles repetidos en la salida canónica de preparación. Sin embargo, la deduplicación puede eliminar clientes distintos con atributos y etiqueta iguales; las matrices versionadas todavía reflejan el flujo anterior hasta regenerarlas.

   **Pendiente:** ejecutar una comparación de sensibilidad con y sin esta regla y reportar cuánto cambian partición, balance y resultados. Verificar luego que train/test generados no contengan perfiles repetidos según la política elegida.

6. **Hay varios flujos de entrenamiento incompatibles.** El notebook 02 ahora usa el CSV de entrada, construye un split estratificado y evalúa pipelines fold-safe; `scripts/run_project.py` genera particiones con `src/split_data.py`, pero `src/train.py` vuelve a leer el CSV original y crea otro split, sin consumir esas particiones. El pipeline de `src` entrena otro SVM con otra configuración, y `main.py` solo entrena. La evaluación del notebook es reproducible, pero aún no hay una ruta canónica común para todos los scripts y artefactos.

7. **El artefacto que el notebook describe como completo no incluye el estimador.** El `model_bundle` guardado contiene columnas y transformadores, pero no incluye `best_model_pipeline` ni el clasificador. Mientras tanto, `scripts/calculate_business_kpis.py` carga un archivo distinto (`support_vector_machine_pipeline.joblib`) y le solicita `predict_proba`. No está demostrado que los artefactos presentes correspondan al último código o sean intercambiables.

   **Corrección para los puntos 6 y 7:** definir un contrato único de entrada/salida; guardar clasificador y todas las transformaciones necesarias en el mismo artefacto; proporcionar un comando oficial de inferencia y una prueba de carga + predicción con datos nuevos.

8. **Las rutas de artefactos siguen divergentes.** Los notebooks guardan gráficos en `images/` y modelos en `models/`; el entrenamiento de `src` también usa `models/`; el script de KPIs carga desde `output/modelos/`. El repositorio mantiene los modelos y gráficos en `output/`, mientras que el README mezcla ambas ubicaciones.

   **Corrección:** centralizar rutas en una raíz de proyecto y usar la misma ruta en notebooks, scripts, documentación y pruebas.

9. **El perfil de clusters ya se generó en la última ejecución del EDA.** `data/processed/cluster_profile_train.csv` está presente en el árbol de trabajo y el notebook de modelado puede leerlo si se requiere. El archivo generado no está versionado actualmente, por lo que una reproducción debe volver a ejecutar el EDA o definir una política de persistencia de datos intermedios.

10. **Las pruebas automatizadas todavía cubren solo una parte pequeña.** Cuatro pruebas verifican normalización/deduplicación, `prepare_dataset`, transformación PCA/K-Means y ajuste/predicción de un pipeline; no prueban split reproducible, persistencia del artefacto ni carga de datos desde el archivo real.

    **Corrección:** extender las pruebas con esquema, transformación, `fit/predict`, carga del artefacto y consistencia del split.

### P2 — Decisión de negocio, explicabilidad y presentación

11. **No hay umbral operativo elegido con costos/capacidad.** Las métricas de clasificación usan el umbral por defecto del estimador, y el KPI de contactar al 20% superior es un ranking separado. El informe recomienda elegir un umbral después, pero no cuantifica capacidad, costos de contacto, falsos positivos y falsos negativos ni una regla final de operación.

    **Corrección:** acordar esos valores con negocio; elegir umbral o tamaño de lista con datos de train/CV y reportar volumen de contactos, churn capturado y errores esperados.

12. **Baseline e incertidumbre CV corregidos en el notebook.** Se añadió `DummyClassifier` y la comparación de candidatos reporta medias y desviaciones entre cinco folds. Sigue siendo una estimación con un solo split externo y no constituye un intervalo de confianza poblacional.

13. **La equidad por grupos está descrita como riesgo, no evaluada.** No se reportan tamaños y errores del modelo por género, edad/tercera edad, contrato u otros grupos relevantes.

    **Corrección:** calcular y discutir Recall, precisión/FPR y tamaños por grupo; explicar límites de la muestra y evitar tratar una diferencia descriptiva como prueba causal.

14. **K-Means continúa siendo exploratorio.** El silhouette `0,0462` indica separación débil. Hay perfil descriptivo, pero no análisis de estabilidad entre semillas/muestras ni evidencia de que el segmento cambie o mejore una acción. No debe presentarse como segmentación operacional validada.

15. **Afirmación causal/financiera corregida.** El README ahora presenta el impacto de campañas como hipótesis y dice explícitamente que las métricas predictivas no demuestran reducción de churn ni aumento de rentabilidad.

    **Corrección:** redactar esos resultados como hipótesis o beneficio potencial y proponer una prueba controlada/seguimiento para medir el efecto real.

16. **El entorno ya puede ejecutarse, pero no está congelado para una reproducción exacta.** Se creó el entorno `telco-ml` desde `environment.yml` y sus imports/dependencias se validaron. `requirements.txt` sigue siendo un manifiesto paralelo, no hay lockfile y todavía falta registrar las versiones efectivamente usadas para la ejecución final. En este equipo, si PowerShell no reconoce `conda`, los comandos se pueden ejecutar desde Miniforge Prompt; el kernel `Python (telco-ml conda)` está registrado y pasó una prueba de humo.
    **Pendiente:** declarar `environment.yml` como fuente oficial, retirar o sincronizar el manifiesto alternativo y congelar/documentar las versiones usadas para la entrega reproducible.

## Estado de los puntos del checklist ampliado

| Sección | Evaluación de esta revisión |
|---|---|
| 01–02 · Problema y EDA | El EDA se ejecutó y generó particiones, perfiles y gráficos. Aún deben conciliarse esos artefactos históricos con el flujo canónico nuevo y analizarse la sensibilidad a la deduplicación. |
| 03 · Preparación y predictores | La importancia por permutación se calculó en un holdout explicativo interno de train. La política de deduplicación está implementada, aunque falta sensibilidad; la selección inicial debe compararse con la relevancia obtenida. |
| 04 · No supervisado | PCA/K-Means y perfiles están documentados; separación débil y utilidad/estabilidad no validadas. |
| 05–06 · Supervisado y decisiones | El notebook actualizado se ejecutó completo: comparación y búsqueda de hiperparámetros con CV en train, baseline, variabilidad entre folds y reporte final del test. Falta umbral comercial cuantificado. |
| 07 · Entrega y defensa | README, guion y guía existen; EDA y notebook 02 se ejecutaron en conda. El subpunto de defensa individual fue aceptado administrativamente, no observado en una defensa. |
| 08 · Validez y leakage | La ejecución limpia verifica test aislado en selección, pipeline dentro de folds, baseline, matriz de confusión con cantidades e incertidumbre CV. Sigue pendiente la selección de umbral/capacidad y la sensibilidad de duplicados. |
| 09 · Reproducibilidad | El notebook 02 se ejecutó completo en conda y guardó sus salidas; cuatro pruebas cubren preparación y pipeline. **Pendiente:** flujo canónico, rutas, fijación de dependencias, artefacto de inferencia y prueba de persistencia. |
| 10 · Negocio, ética y monitoreo | **Pendiente:** capacidad/costos, umbral, métricas por grupos, validación útil de clusters, población/fecha de aplicación y plan medible de campaña. |

## Orden recomendado de trabajo

1. Comparar resultados con y sin la regla de perfiles repetidos y documentar su sensibilidad.
2. Definir un flujo canónico compartido por notebook, scripts y artefacto; alinear rutas de gráficos/modelos.
3. Guardar el modelo seleccionado y probar carga/inferencia con datos nuevos.
4. Definir capacidad, costos y umbral operativo sin reajustar el modelo sobre el test ya reportado; usar nueva evaluación independiente si se modifica.
5. Completar métricas de errores por segmentos y validar utilidad/estabilidad de clusters.
6. Revisar salidas EDA antiguas, fijar dependencias y documentar límites de población/fecha.
7. Cerrar el checklist restante solo con evidencia reproducible.
