# Presentación EP2 · Telco Customer Churn

Este documento cumple tres funciones: estructura de diapositivas, guion hablado de 10 minutos para tres integrantes y especificación para generar una primera PPT con una IA. La IA debe apoyar el diseño, no inventar resultados ni sustituir la revisión del equipo.

## 1. Reglas y narrativa común

- **Duración total:** máximo 10 minutos. Objetivo de ensayo: 9:30–9:45 para dejar margen a pausas.
- **Orden sin alternancia:** Integrante 1 presenta consecutivamente la primera tercera parte (0:00–3:20); Integrante 2 continúa con la segunda tercera parte (3:20–6:40); Integrante 3 cierra con la última tercera parte (6:40–10:00).
- **Reparto sugerido:** 9 diapositivas, tres por integrante. Ninguna persona vuelve a hablar hasta las preguntas.
- **Hilo narrativo:** problema y datos → análisis y método → resultados, decisión, límites y cierre.
- **Tono:** claro para audiencia administrativa y técnicamente correcto para una clase de Machine Learning.
- **Fuente de verdad:** [README](../README.md), [notebook EDA](../notebooks/01_EDA_y_Limpieza.ipynb), [notebook de modelado](../notebooks/02_Modelado_y_Evaluacion.ipynb) y reportes de `output/resultados/`.
- El score SVM es un `decision_function`, **no** una probabilidad calibrada.
- No describir asociación como causalidad ni el escenario económico como ahorro observado.
- No decir que el test es completamente ciego: la selección programática no lo usa antes del reporte, pero el EDA exploró la cohorte completa.

## 2. Guion por diapositiva y por integrante

### Integrante 1 · Contexto, datos y hallazgos (0:00–3:20)

#### Diapositiva 1 · Del problema a una pregunta medible (0:00–1:05)

**En pantalla**

- Telco Customer Churn
- Pregunta: ¿cómo priorizar clientes para una revisión preventiva?
- Objetivo: ordenar por riesgo; no automatizar ofertas

**Guion sugerido**

“Nuestro proyecto aborda el abandono de clientes de telecomunicaciones. En lugar de prometer que podemos evitarlo, formulamos una pregunta medible: ¿podemos ordenar los clientes según su riesgo histórico para que un equipo decida a quién revisar primero? Es un problema de clasificación supervisada. El resultado es un score de decisión para apoyar a personas; no es una probabilidad calibrada, una campaña ni una decisión automática.”

**Visual:** flujo simple Cliente/datos → modelo → lista priorizada → revisión humana. No usar un gráfico que sugiera una campaña ya ejecutada.

#### Diapositiva 2 · Datos y calidad (1:05–2:10)

**En pantalla**

- 7.043 registros · 19 predictores · objetivo `Churn`
- 1.869 churn / tasa histórica 26,54%
- `TotalCharges`: texto con 11 espacios vacíos; conversión y tratamiento documentados
- 22 perfiles repetidos tras normalización; no prueban identidad

**Guion sugerido**

“Usamos el dataset público Telco Customer Churn con 7.043 registros y 19 predictores originales. La etiqueta es `Churn`, y 1.869 clientes, un 26,54%, aparecen como abandono. Revisamos tipos, vacíos y perfiles repetidos: `TotalCharges` venía como texto y tenía 11 espacios en blanco asociados a antigüedad cero. También encontramos perfiles iguales al ignorar el identificador, pero eso no demuestra que sean la misma persona; evaluamos la sensibilidad a conservarlos o deduplicarlos.”

**Visual:** una tarjeta de conteos o barra de proporción 26,54%/73,46%.

#### Diapositiva 3 · Qué aprendimos del EDA (2:10–3:20)

**En pantalla**

- Más churn descriptivo en contratos mes a mes y menor antigüedad
- `tenure`–`TotalCharges`: correlación 0,826
- Hallazgos asociativos, no causales

**Guion sugerido**

“El análisis exploratorio muestra mayor abandono histórico en contratos mes a mes y entre clientes con menos antigüedad. `tenure` y `TotalCharges` tienen correlación 0,826, algo razonable porque el total facturado tiende a crecer con el tiempo. La correlación describe asociación, no prueba que una variable cause la otra. Además, el EDA examinó toda la cohorte, incluida la etiqueta; lo reconocemos como una limitación de la independencia humana del test.”

**Transición (últimos segundos):** “Con esos datos y límites definidos, ahora explicamos cómo preparamos y comparamos los modelos.”

### Integrante 2 · Preparación, evaluación y modelos (3:20–6:40)

#### Diapositiva 4 · Cómo evitamos fuga de información (3:20–4:25)

**En pantalla**

- Train → CV de 5 folds → test reservado
- Transformaciones y SMOTE dentro del pipeline/fold
- Test solo para el reporte final

**Guion sugerido**

“Separamos entrenamiento y test de forma estratificada. La selección y ajuste se hacen dentro de entrenamiento con cinco folds. Imputación, escalado, codificación, PCA/K-Means cuando corresponde y SMOTE viven dentro del pipeline, así que cada fold aprende sus transformaciones solo de su parte de entrenamiento. El test no selecciona el ganador ni los hiperparámetros. Como el EDA sí exploró la cohorte completa, no afirmamos que el test sea totalmente ciego frente a decisiones humanas; para confirmación necesitamos datos nuevos.”

**Visual:** esquema de CV en el que un fold queda como validación y las operaciones se ajustan dentro del fold de entrenamiento.

#### Diapositiva 5 · Modelos y criterio de selección (4:25–5:30)

**En pantalla**

- Dummy baseline, Regresión Logística, Random Forest, Gradient Boosting, SVM
- Optuna ajusta SVM en CV de entrenamiento
- Recall como prioridad; ROC-AUC como lectura complementaria

**Guion sugerido**

“Comparamos una referencia simple y cuatro algoritmos supervisados: Regresión Logística, Random Forest, Gradient Boosting y SVM. También evaluamos una representación PCA más K-Means. Se eligió el SVM ajustado con Optuna usando Recall en validación cruzada de entrenamiento; ROC-AUC ayuda a evaluar el ordenamiento en distintos umbrales. No elegimos por el resultado del test.”

**Visual:** gráfico de resultados CV de los modelos, si se usa, tomado del notebook vigente; etiquetar media y desviación, no solo una barra sin escala.

#### Diapositiva 6 · Resultado técnico y K-Means (5:30–6:40)

**En pantalla**

- SVM: Recall CV 0,8007 ± 0,0388; ROC-AUC CV 0,8471 ± 0,0186
- PCA + K-Means: Recall CV 0,7751
- Silhouette K-Means: 0,0462 → exploratorio, no operativo

**Guion sugerido**

“El SVM original obtiene Recall CV de 0,8007 con desviación 0,0388 y ROC-AUC de 0,8471 con desviación 0,0186. La alternativa con PCA y K-Means alcanza Recall 0,7751, por debajo del modelo original, y silhouette 0,0462 indica separación débil. K-Means agrupa por similitud, no predice churn individual; por eso queda como análisis exploratorio y no define a quién contactar.”

**Transición:** “Ya explicado el método, la última parte muestra el test, la interpretación operativa y qué falta antes de usar el modelo en un negocio.”

### Integrante 3 · Resultados, límites y próximos pasos (6:40–10:00)

#### Diapositiva 7 · Evaluación en test (6:40–7:45)

**En pantalla**

- Recall 0,7715 (287/372)
- Precisión 0,5116 · F1 0,62 · ROC-AUC 0,8405
- Matriz: TN 759 · FP 274 · FN 85 · TP 287
- Top 20%: captura 192/372 = 51,61% (281 clientes)

**Guion sugerido**

“En el test reservado, el Recall fue 0,7715: detectamos 287 de los 372 abandonos reales al umbral de clasificación reportado. La precisión fue 0,5116, así que también hay falsos positivos: 274 clientes marcados que no abandonaron en este test. La matriz completa permite ver ambos tipos de error. Si se limita la lista al 20% superior por score, 281 clientes concentran 192 de los 372 churn. Eso evalúa priorización histórica, no significa que retengamos a esos clientes.”

**Visual:** matriz de confusión con conteos o dos visuales simples. Si se combina con captura top 20%, identificar que representa otro criterio/escenario.

#### Diapositiva 8 · Escenario hipotético y errores por grupos (7:45–8:55)

**En pantalla**

- Supuestos: 250/7.043 contactos · eficacia 20% · horizonte 24 meses
- Costo 1/5/10% de mediana `MonthlyCharges`; facturación bruta proxy
- Escenario test: 44 contactos, 40 TP, 8 retenciones supuestas
- Grupos: contrato mes a mes Recall 0,864 / FPR 0,600; dos años Recall 0 con 8 churn

**Guion sugerido**

“Simulamos, no observamos, una política con capacidad de 250 contactos por cada 7.043 clientes, eficacia supuesta de 20% y horizonte de 24 meses. `MonthlyCharges` se usa como facturación bruta proxy, no como margen ni utilidad. El escenario test prioriza 44 clientes, incluye 40 churn reales y calcula ocho retenciones solo al aplicar el supuesto del 20%. Por grupo, el contrato mes a mes presenta Recall 0,864 pero FPR 0,600; el contrato de dos años tiene Recall cero con solo ocho churn, por lo que es una estimación incierta. No es una prueba automática de sesgo.”

**Visual:** tarjeta de “supuesto vs. dato medido” o tabla corta con dos grupos; evitar gráfico con ejes o intervalos omitidos.

#### Diapositiva 9 · Conclusión, límites y decisión (8:55–10:00)

**En pantalla**

- Entrega: pipeline reproducible + score de priorización
- No hay umbral comercial aprobado ni impacto causal medido
- Siguiente paso: cohorte independiente + costos reales + piloto controlado

**Guion sugerido**

“La conclusión es que construimos un flujo reproducible que permite ordenar clientes y evaluar sus errores con métricas técnicas. No aprobamos una política comercial: el umbral operativo del artefacto permanece vacío, los costos son supuestos y el ahorro no es observado. El test requiere confirmación en una cohorte futura porque el EDA exploró toda la muestra. Antes de operar, proponemos validar con datos económicos reales y medir la intervención con un piloto controlado. Gracias; estamos preparados para preguntas.”

## 3. Prompt listo para una IA generadora de presentaciones

Copien el texto siguiente en una herramienta de generación de presentaciones. Adjuntar esta guía y, si la herramienta permite archivos, el README o las capturas/figuras indicadas. No subir datos personales ni información privada.

> Crea una presentación académica en español para una evaluación de Machine Learning sobre el proyecto Telco Customer Churn. Debe tener exactamente 9 diapositivas y durar como máximo 10 minutos. Audiencia: docente analista de datos y estudiantes de analítica. Sigue estrictamente el guion y las cifras del documento fuente adjunto; no inventes números, afirmaciones, citas, causalidad ni resultados.
>
> Divide la exposición en tres bloques consecutivos, sin alternar expositores: diapositivas 1–3 las presenta Integrante 1 (0:00–3:20); diapositivas 4–6 Integrante 2 (3:20–6:40); diapositivas 7–9 Integrante 3 (6:40–10:00). Incluye en las notas del orador el texto hablado sugerido y la duración indicada para cada diapositiva.
>
> Narrativa: problema y datos; EDA y límites; prevención de fuga de información; comparación de modelos; resultados de test; escenario económico explícitamente hipotético; errores por grupos; próximos pasos. Destaca que el score SVM no es probabilidad calibrada, que el EDA exploró la cohorte completa, y que ni la simulación económica ni las métricas predictivas prueban un efecto causal.
>
> Cifras autorizadas: 7.043 registros, 19 predictores originales, 1.869 churn, churn histórico 26,54%; correlación `tenure`–`TotalCharges` 0,826; SVM Recall CV 0,8007 ± 0,0388, ROC-AUC CV 0,8471 ± 0,0186; SVM PCA + K-Means Recall CV 0,7751; silhouette 0,0462; test Recall 0,7715 (287/372), precisión 0,5116, F1 0,62, ROC-AUC 0,8405; matriz TN 759, FP 274, FN 85, TP 287; captura top 20% 192/372 = 51,61%, 281 clientes; escenario hipotético 250 contactos/7.043, eficacia supuesta 20%, horizonte 24 meses, costo de contacto 1/5/10% de la mediana mensual, escenario test 44 contactos/40 TP y 8 retenciones calculadas bajo el supuesto; contrato mes a mes Recall 0,864/FPR 0,600 y contrato de dos años Recall 0 con solo 8 churn. No presentar montos como moneda o utilidad.
>
> Diseño: 16:9, sobrio, alto contraste, tipografía legible, máximo 4–5 viñetas cortas por diapositiva. Utiliza diagramas simples para el flujo train/CV/test y la ruta score→revisión humana; usa matriz de confusión y un gráfico real solo si está entre los archivos suministrados. Si una imagen no está adjunta, deja un espacio claramente rotulado con el nombre de la figura a insertar; no generes gráficos ni datos ficticios. Coloca fuente y nota de cautela en cada diapositiva con resultados.
>
> Entrega también notas de orador con transiciones entre integrantes. No agregues una décima diapositiva de contenido; una portada debe contar como diapositiva 1. Evita animaciones, párrafos extensos, decoración que compita con los datos y afirmaciones de despliegue productivo. Al final, entrega un resumen de cualquier cifra o dato que no hayas podido verificar para revisión manual.

## 4. Cómo revisar y terminar la PPT generada

### Revisión factual

- Comparen cada número con README/notebooks; revisen especialmente decimales, numeradores/denominadores y signos ±.
- Comprueben que CV sea presentado como media ± desviación entre folds y test como evaluación final del modelo, no como validación independiente de cualquier decisión humana.
- Verifiquen que el score no aparezca como “probabilidad”.
- Diferencien resultados del clasificador al umbral por defecto, captura top 20% y escenario económico OOF/test: son análisis distintos.
- Señalen los supuestos de eficacia, capacidad, horizonte y costo; no inventen moneda, margen, CLV o costo de campaña.
- Asegúrense de que los hallazgos por grupo se describan como diferencias post hoc con conteos e incertidumbre, no como certificación de equidad.

### Revisión visual y de accesibilidad

- Mantengan una idea central por diapositiva; reduzcan texto antes que reducir demasiado la fuente.
- Revisen que las matrices y gráficos tengan ejes, unidades, leyenda y tamaño de muestra.
- Usen color y etiquetas redundantes; no dependan solo de rojo/verde.
- Comprueben tildes, nombres (`Churn`, `MonthlyCharges`, SVM, K-Means) y legibilidad en pantalla compartida.
- Revisen contraste, alineación, márgenes, consistencia de títulos y que ninguna figura esté pixelada o cortada.
- Eliminen visuales decorativos generados por IA que puedan parecer evidencia real.

### Revisión del guion y del tiempo

- Lean en voz alta las notas; no intenten decir cada viñeta literalmente.
- Cronometren cada bloque: 3:20, 3:20 y 3:20 como máximo.
- Añadan transiciones cortas; no repitan la explicación del problema ni los mismos resultados entre integrantes.
- Practiquen el cambio de expositor y asignen quién responde primero preguntas; todos deben poder defender el proyecto completo.
- Exporten a PDF y prueben el archivo en otro equipo o visor. Comprueben fuentes, gráficos y orden de páginas.
- Guarden una copia final editable y PDF. No ejecuten Optuna ni dependan de conexión a internet durante la defensa.

## 5. Banco de preguntas probables y respuestas modelo

Respuestas modelo para estudiar, no para recitar. Si el profesor pide detalle, expliquen razonamiento, evidencia y limitación.

### Problema, datos y EDA

**1. ¿Cuál es el problema de negocio y quién usa la salida?**
El problema es priorizar clientes con riesgo histórico de churn para que un equipo revise posibles acciones preventivas. La salida es un score de ordenamiento para apoyar a una persona; no automatiza la oferta ni demuestra que contactar al cliente evite su abandono.

**2. ¿Por qué es clasificación y no regresión?**
La etiqueta objetivo `Churn` tiene dos clases, `Yes` y `No`; buscamos estimar/ordenar la clase de abandono, no predecir un valor continuo. El score del SVM no está calibrado como probabilidad.

**3. ¿Qué controles de calidad fueron importantes?**
`TotalCharges` venía como texto y tenía 11 espacios vacíos asociados a `tenure=0`; se convirtió y trató conforme a la regla documentada. También se identificaron perfiles repetidos ignorando ID, pero no se afirma que correspondan a la misma persona.

**4. ¿La correlación 0,826 entre antigüedad y cargos totales implica causalidad?**
No. Es una asociación numérica fuerte, compatible con la acumulación de cargos al permanecer más tiempo. No demuestra mecanismo causal ni justifica por sí sola eliminar una variable.

### Preparación, validación y selección

**5. ¿Cómo controlaron la fuga de información?**
La comparación y ajuste se realizan dentro de train con CV. Las transformaciones aprendidas y SMOTE están dentro del pipeline, por lo que cada fold aprende solo de su fold de entrenamiento. El test se usa después de cerrar la selección. Reconocemos que el EDA sí exploró la cohorte completa, así que una cohorte futura independiente sería mejor confirmación.

**6. ¿Qué diferencia hay entre train, CV y test?**
Train ajusta los modelos. CV divide train en folds para comparar y ajustar sin usar test. Test ofrece una evaluación final bajo el diseño establecido, pero no debe reutilizarse para cambiar decisiones. Aquí su independencia humana está limitada por el EDA previo.

**7. ¿Por qué SMOTE no se aplica antes de separar?**
Aplicarlo antes permitiría que información de ejemplos sintéticos relacionados llegara a validación/test y alteraría la prevalencia. Se ejecuta solo dentro de entrenamiento de cada fold.

**8. ¿Por qué usar Recall y ROC-AUC?**
Recall mide cuántos abandonos positivos detectamos en un umbral. ROC-AUC mide ordenamiento global sobre distintos umbrales. Recall refleja la prioridad de no dejar muchos churn sin marcar, pero debe leerse junto a precisión, FPR y capacidad porque puede aumentar contactos equivocados.

**9. ¿Cómo interpretan la desviación de Recall CV?**
El 0,0388 es la desviación observada entre los cinco folds, no un intervalo de confianza poblacional. Señala variación del rendimiento según la partición; no prueba estabilidad futura por sí sola.

**10. ¿Por qué el SVM no entrega probabilidad?**
El flujo usa `decision_function`, un score de margen/ranking. No se calibró con un método probabilístico, así que no debe comunicarse como “hay 80% de probabilidad de churn”.

### Resultados, grupos y no supervisado

**11. ¿Qué significa Recall test 0,7715?**
En el umbral reportado, se identificaron 287 de los 372 churn reales del test; quedaron 85 falsos negativos. No significa que 77% de esos clientes se retendrían.

**12. ¿Qué implican 274 falsos positivos?**
Son clientes clasificados como churn que no abandonaron en ese test. Pueden consumir capacidad de revisión/contacto. Su costo real no está medido; por eso Recall debe leerse con precisión 0,5116 y con una capacidad de negocio validada.

**13. ¿Qué aporta capturar 51,61% de churn en el top 20%?**
En el test, ordenar y elegir 281 clientes de mayor score incluye 192 de los 372 churn. Es útil para ilustrar priorización limitada, pero no equivale a retenerlos ni es el umbral económico del otro escenario.

**14. ¿Por qué no usar K-Means para contactar?**
K-Means agrupa similitud, no aprende la etiqueta churn. La separación fue débil (silhouette 0,0462) y la variante SVM con PCA + K-Means obtuvo menor Recall CV, 0,7751 frente a 0,8007. Se conserva como exploración/comparación.

**15. ¿Las diferencias de Recall por contrato demuestran discriminación?**
No. Son tasas descriptivas en la evaluación disponible; el contrato de dos años tiene solo ocho churn y su estimación es incierta. Para concluir sobre equidad se necesitarían criterios definidos, análisis con soporte adecuado, evaluación contextual y datos no explorados; no se retocó el modelo con estas cifras.

### Política y límites

**16. ¿Cómo se eligió el umbral 1,7907 del escenario?**
Se eligió con scores `decision_function` out-of-fold del entrenamiento, minimizando costo proxy esperado sujeto a la capacidad hipotética. No es umbral de probabilidad ni umbral operativo aprobado; el artefacto mantiene el umbral operativo vacío.

**17. ¿El ahorro de 13.207–13.486 unidades es real?**
No. Es un cálculo hipotético con capacidad, eficacia del 20%, horizonte de 24 meses y costo de contacto asumidos. `MonthlyCharges` aproxima facturación bruta; no es margen, utilidad o CLV. No hubo intervención, grupo control ni medición causal.

**18. ¿Cómo comprobarían que una campaña funciona?**
Definirían población elegible, tratamiento, costos y resultado con el área comercial. Idealmente asignarían aleatoriamente una campaña y un grupo de control, medirían retención incremental, costos, margen y posibles daños, y reportarían incertidumbre. Sin ese diseño, la predicción de riesgo no identifica el efecto de la intervención.

**19. ¿Qué falta antes de producción?**
Validar en una cohorte temporal/externa no explorada; conseguir datos reales de capacidad, margen y costos; acordar el umbral y reglas humanas; probar integración, calidad de datos, monitoreo y desempeño por grupos; medir la eficacia de campañas con control. El proyecto es académico, no un sistema productivo aprobado.

**20. ¿Qué harían si una nueva cohorte baja el rendimiento?**
Primero revisarían integridad y deriva de datos, prevalencia, esquema y métricas por grupo; compararían con una línea base y cuantificarían incertidumbre. Se diagnosticaría y documentaría una nueva iteración usando una estrategia de validación apropiada, manteniendo un conjunto de confirmación nuevo sin reutilizar el test actual para ajustes sucesivos.

## 6. Cierre de equipo

La exposición está lista para ensayarse cuando los tres integrantes pueden explicar sin leer su tercio, responder preguntas de cualquier sección y abrir la evidencia que respalda sus afirmaciones. La actualización de estos documentos prepara el ensayo, pero la práctica y el tiempo solo quedan verificados después de realizar y registrar el ensayo real.
