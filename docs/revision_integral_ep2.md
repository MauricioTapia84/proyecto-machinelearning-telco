# Revisión integral del proyecto EP2

**Base de la revisión:** commit `14d4b76` (`3.5. entrega y defensa refinadas`), con avances de corrección posteriores en el árbol de trabajo.

**Alcance:** README, checklist, changelog, documentación de defensa, notebooks, scripts, código reutilizable, datos procesados, rutas y artefactos presentes.

## Dictamen ejecutivo

La documentación de negocio explica problema, KPIs, CRISP-DM, modelos, hallazgos de EDA y límites de uso. El notebook comparte selección, pipeline y persistencia con los scripts; también cubre sensibilidad a perfiles repetidos, permutation importance y SHAP del score actual. Se documentó una política económica hipotética, métricas de error por grupo, una decisión de no operar K-Means y un lock reproducible de Windows. La guía didáctica, el guion consecutivo para tres integrantes, las instrucciones de creación/revisión de una PPT con IA y el banco de preguntas ya están preparados; generar y cronometrar la PPT y verificar la defensa individual siguen pendientes.

**Prioridad restante antes de una campaña real:** reemplazar los supuestos por capacidad y economía observadas, medir eficacia mediante piloto controlado y confirmar resultados con una cohorte futura no explorada. La simulación no equivale a rentabilidad y el umbral sigue sin incorporarse al artefacto.

El entorno validado usa Python 3.11.17; `environment.yml` fija las dependencias directas y `environment-win-64.lock.txt` registra la resolución explícita de Windows x64. El lock y la declaración YAML pasaron resolución en modo dry-run. Doce pruebas automatizadas pasan. La comparación CV seleccionó SVM ajustado (Recall 0,8007 ± 0,0388; ROC-AUC 0,8471 ± 0,0186); test obtuvo Recall 0,7715 y ROC-AUC 0,8405. El script canónico guardó el pipeline bajo `output/modelos/`; los scripts de política y grupos se ejecutaron correctamente y escribieron sus reportes en `output/resultados/`.

**Límite importante de la evaluación:** la selección y el cálculo de métricas no consultan el test hasta el reporte final, pero varias visualizaciones del EDA examinan la cohorte completa, incluida la etiqueta. Por tanto, el test está aislado del código de selección, pero no es completamente ciego frente a decisiones humanas informadas por la exploración; la confirmación requiere una cohorte nueva o temporal no explorada.

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

5. **La regla de perfiles repetidos no confirma identidad.** `src.data.normalize_customer_profiles` conserva todas las filas y normaliza los atributos que definen un perfil; `deduplicate_customer_profiles` excluye `customerID` e incluye `Churn`. El EDA documenta 22 perfiles repetidos eliminados de 7.043 filas. `src/sensitivity.py` y el notebook 02 compararon las políticas solo en entrenamiento; el test y sus copias quedaron fuera, y la CV sin deduplicar agrupó cada perfil en un fold. Se obtuvieron Recall 0,7940 ± 0,0282 / ROC-AUC 0,8474 ± 0,0112 con repeticiones frente a Recall 0,7987 ± 0,0287 / ROC-AUC 0,8448 ± 0,0111 deduplicando. Las diferencias son pequeñas frente a la desviación entre folds; la justificación para deduplicar es evitar el sobrepeso de perfiles iguales, no identificar clientes ni mejorar materialmente el modelo. Una CV ingenua repartiría 17 perfiles repetidos entre folds.

6. **Flujo de entrenamiento unificado y ejecutado.** `src/modeling.py` comparte la comparación, Optuna y la selección entre el notebook 02 y `src/train.py`. `scripts/run_project.py` ejecuta ese flujo sin generar una partición alternativa antes. La ejecución canónica terminó con código 0 y eligió SVM ajustado (Optuna). `src/split_data.py` queda como herramienta auxiliar del EDA, no como parte del entrenamiento canónico.

7. **Artefacto e inferencia unificados y comprobados.** El notebook y el entrenamiento guardan el pipeline ajustado completo con su metadata en `output/modelos/telco_churn_pipeline.joblib`. `src/inference.py` valida el formato, revisa predictores requeridos y expone `score_customers`; el script de KPIs cargó el artefacto generado por el comando canónico y calculó predicciones y scores, sin exigir `predict_proba` a un SVM.

8. **Rutas consolidadas y verificadas.** `src/paths.py` establece las ubicaciones oficiales. Los notebooks y scripts escriben los gráficos/modelos bajo `output/graficos/` y `output/modelos/`. Los artefactos que seguían en `images/` y `models/` se trasladaron a `output/`; los nombres que colisionaban con otros archivos se conservaron con sufijo `_legacy`. El EDA se volvió a ejecutar tras el cambio: sus 29 celdas de código terminaron sin errores, generó gráficos/modelos bajo `output/` y las carpetas anteriores `images/` y `models/` quedaron vacías.

9. **El perfil de clusters ya se generó en la última ejecución del EDA.** `data/processed/cluster_profile_train.csv` está presente en el árbol de trabajo y el notebook de modelado puede leerlo si se requiere. El archivo generado no está versionado actualmente, por lo que una reproducción debe volver a ejecutar el EDA o definir una política de persistencia de datos intermedios.

10. **Cobertura automatizada ampliada, aún parcial.** Doce pruebas verifican normalización/deduplicación, preparación, PCA/K-Means, ajuste, persistencia/inferencia, sensibilidad a duplicados, economía del escenario, elección de umbral y métricas por grupos. No sustituyen validación con lotes de producción, confirmación temporal independiente ni una prueba de campaña.

### P2 — Decisión de negocio, explicabilidad y presentación

11. **Política operativa hipotética calculada; política comercial real pendiente.** `src/decision.py` minimiza costo esperado sujeto a capacidad con scores OOF de entrenamiento. Con 250 contactos/7.043, eficacia supuesta 20%, horizonte 24 meses, costos 1/5/10% de la mediana mensual y pérdida proxy de facturación bruta, los escenarios seleccionan umbral 1,7907. En test se seleccionan 44/50; la simulación estima pérdida base de 628.531,20 unidades, pérdida neta de 615.045,38–615.324,16 y ahorro de 13.207,04–13.485,82 unidades. El artefacto conserva `operating_threshold=None`. No se conocen moneda, margen, costo de oferta, eficacia o capacidad reales; las cifras son escenarios y no estimaciones de rentabilidad ni efecto observado.

12. **Baseline e incertidumbre CV corregidos en el notebook.** Se añadió `DummyClassifier` y la comparación de candidatos reporta medias y desviaciones entre cinco folds. Sigue siendo una estimación con un solo split externo y no constituye un intervalo de confianza poblacional.

13. **Métricas de error por grupos calculadas, sin constituir certificación de equidad.** `scripts/evaluate_group_metrics.py` reporta conteos, Recall, precisión, FPR, tasas de selección e intervalos Wilson en género, tercera edad, contrato, internet y antigüedad. En test, `SeniorCitizen=1` presenta Recall 0,910/FPR 0,493 frente a 0,728/0,230 para `SeniorCitizen=0`; contratos mes a mes presentan Recall 0,864 y FPR 0,600; contratos de dos años tienen solo 8 churn y Recall 0. Las diferencias ameritan revisión; las bases pequeñas y la exploración previa del test impiden concluir que exista o no sesgo. No se ajustó el modelo a partir de estos resultados.

14. **Decisión sobre K-Means cerrada para el alcance del proyecto.** Se conserva solo como comparación metodológica y exploración, no como regla operativa. La silhouette 0,0462 refleja separación débil y el SVM con PCA/K-Means obtuvo Recall CV 0,7751 frente a 0,8007 con predictores originales. No se justifica invertir en estabilidad de segmentos si no se usarán para asignar acciones; esa evaluación sería necesaria si el alcance cambiara a segmentación comercial.

15. **Afirmación causal/financiera corregida.** El README ahora presenta el impacto de campañas como hipótesis y dice explícitamente que las métricas predictivas no demuestran reducción de churn ni aumento de rentabilidad.

    **Corrección:** redactar esos resultados como hipótesis o beneficio potencial y proponer una prueba controlada/seguimiento para medir el efecto real.

16. **Entorno fijado y lock validado.** `environment.yml` contiene versiones exactas de dependencias directas y usa únicamente conda-forge; `environment-win-64.lock.txt` fija las 240 entradas explícitas (incluidas dependencias transitivas/builds) de Windows x64. Se retiró el `requirements.txt` duplicado. La resolución del lock explícito y el YAML pasó `conda create/conda env create --dry-run`. Versiones clave: Python 3.11.17, pandas 3.0.6, NumPy 2.4.6, scikit-learn 1.9.1, imbalanced-learn 0.14.2, SHAP 0.51.0 y Optuna 5.0.0.

## Estado de los puntos del checklist ampliado

| Sección | Evaluación de esta revisión |
|---|---|
| 01–02 · Problema y EDA | El EDA se volvió a ejecutar tras consolidar rutas: 29/29 celdas de código, sin errores, con gráficos/modelos en `output/`. Se documentaron correlaciones (incluido que `tenure`–`TotalCharges` = 0,826), tasas categóricas y que no se ejecutó chi-cuadrado ni se eliminaron columnas por > 0,95. La sensibilidad a perfiles repetidos se midió solo en entrenamiento. El EDA completo limita la ceguera humana del test. |
| 03 · Preparación y predictores | La importancia por permutación se calculó en un holdout explicativo interno de train y se interpretó frente a los 19 predictores originales; no se usa para re-seleccionar el modelo. La sensibilidad de deduplicación está completada. |
| 04 · No supervisado | K-Means queda como comparación exploratoria y no se usa operativamente: silhouette débil y menor Recall que la representación original. |
| 05–06 · Supervisado y decisiones | CV y evaluación final están ejecutadas. La política económica hipotética calcula pérdida evitada, costo de contacto, pérdida residual y ahorro proxy, manteniendo explícitos sus supuestos. La política real sigue pendiente de costos y capacidad observados. |
| 07 · Entrega y defensa | La guía didáctica, el guion consecutivo para tres integrantes, las instrucciones para PPT con IA y el banco de 20 preguntas están documentados. Falta generar/revisar/cronometrar la PPT final y realizar/registrar el ensayo real. Los notebooks y scripts se ejecutaron con Conda. |
| 08 · Validez y leakage | La ejecución verifica que la selección programática no consulta test, que el pipeline se ajusta dentro de folds, y que hay baseline, matriz de confusión con cantidades e incertidumbre CV. El EDA sí inspecciona la cohorte completa, así que el test no es una muestra ciega para decisiones humanas. Sigue pendiente una confirmación con datos independientes y una política comercial. |
| 09 · Reproducibilidad | Se fijaron dependencias directas, lock Windows x64 y 12 pruebas; ambas definiciones de entorno pasaron dry-run. Los scripts de escenarios y métricas por grupos se ejecutaron y guardaron resultados en `output/resultados/`. |
| 10 · Negocio, ética y monitoreo | Métricas por grupos y decisión no operativa de K-Means documentadas. Siguen pendientes umbral/costos reales, cohorte futura independiente, validación de población y medición de eficacia con campaña controlada. |

## Orden recomendado de trabajo

1. Generar la PPT con las instrucciones de `docs/presentacion_ep2.md`, cotejar cada lámina con README/notebooks y exportar una copia PDF.
2. Ensayar y registrar la exposición grupal (tres bloques consecutivos, máximo 10 minutos) y comprobar la defensa individual de cada integrante usando la guía y el banco de preguntas.
3. Para cualquier uso operativo futuro, obtener capacidad, margen/costos, valor de cliente y eficacia de contacto; recalcular el umbral OOF sin tratar los proxies como datos reales.
4. Evaluar el modelo en una cohorte futura no explorada y repetir el análisis por grupo antes de uso operativo.
5. Medir el efecto incremental de una intervención mediante piloto controlado o seguimiento comparable.
6. Mantener el lock alineado si se cambian dependencias; conciliar artefactos históricos `_legacy` solo si es necesario para la entrega.
