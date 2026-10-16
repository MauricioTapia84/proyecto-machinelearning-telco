CHECKLIST DE REVISIÓN · EP2 MACHINE LEARNING
Marcar cada punto al revisar el proyecto. Basado en la rúbrica, la transcripción del profesor y los apuntes de Mauricio.
Marca cada punto cuando haya evidencia en el código o la documentación; la ejecución reproducible de esa evidencia se verifica por separado en la sección 09.

01 · Problema y metodología
☑  Problema de negocio, objetivos y KPIs definidos.
☑  Fuente de datos y variable objetivo identificadas.
☑  Etapas de CRISP-DM documentadas.
☑  Impacto ético del proyecto considerado.

02 · Análisis exploratorio (EDA)
☑  Calidad de datos revisada: tipos, nulos, duplicados e inconsistencias.
☑  Cada variable tiene análisis individual, gráfico y estadísticas pertinentes.
☑  Distribuciones y patrones relevantes interpretados.
☑  Variables predictoras comparadas con la variable objetivo.
☑  Relaciones entre predictores analizadas (gráficos cruzados y correlación).
☑  Análisis multivariable realizado y hallazgos explicados.

03 · Preparación y predictores
☑  Limpieza y transformaciones aplicadas y justificadas.
☑  Datos separados para entrenamiento y prueba, sin fuga de información en validación cruzada (ejecutado en el notebook de modelado).
☑  Predictores iniciales seleccionados y justificados antes del modelamiento.
☐  Relevancia de predictores revisada después y comparada con la selección inicial sin usar el test para decidir.
☑  Uso de PCA evaluado frente a variables originales mediante una evaluación sin fuga en CV.

04 · Aprendizaje no supervisado
☑  Al menos una técnica no supervisada implementada.
☑  Patrones, agrupaciones o segmentos identificados e interpretados.
☑  Utilidad de los hallazgos para el negocio explicada.

05 · Modelos supervisados
☑  Problema definido como clasificación o regresión.
☑  Al menos dos modelos supervisados implementados.
☑  Selección de algoritmos y configuraciones justificada.
☑  Modelos comparados con métricas adecuadas y elección final explicada a partir de validación sin fuga.

06 · Resultados y decisiones
☑  Métricas interpretadas y sincronizadas con una ejecución limpia.
☑  Conclusiones y recomendaciones vinculadas a KPIs reproducidos en el notebook/README.
☐  Explicado cómo se usaría el modelo y qué decisión apoyaría, con capacidad/umbral definidos.

07 · Entrega y defensa
☑  README.md incluye problema, objetivos, KPIs, datos, EDA y CRISP-DM.
☑  Los notebooks EDA y modelado se ejecutaron de principio a fin con el entorno conda.
☐  Datasets y carpetas del proyecto están organizados y permiten reproducir el flujo canónico.
☑  Presentación cubre los hallazgos clave en un máximo de 10 minutos.
☐  Cada integrante puede defender cualquier parte del proyecto (pendiente de práctica/verificación).

08 · Validez de la evaluación y fuga de información
☑  El conjunto de prueba final no se usó para elegir modelo ni ajustar hiperparámetros; el umbral operativo queda pendiente.
☑  La selección de modelos e hiperparámetros se realizó dentro de entrenamiento mediante validación cruzada estratificada.
☑  Imputación, escalado, codificación, PCA/K-Means y balanceo se ajustaron dentro de cada pliegue.
☑  Se definió y reportó un modelo base de comparación.
☑  Se reportó la matriz de confusión en cantidades, además de proporciones y métricas.
☐  Se analizaron precisión-recall y distintos umbrales; el umbral final tiene justificación operativa.
☑  Se reportó variabilidad/incertidumbre de las métricas entre pliegues.
☐  Se justificó la política de duplicados y se comparó el impacto con/sin eliminación de perfiles repetidos.

09 · Reproducibilidad e implementación
☐  Hay un único flujo oficial desde los datos de entrada hasta el modelo guardado.
☐  El comando descrito en el README ejecuta realmente ese flujo.
☐  Los notebooks se ejecutaron desde un kernel limpio del entorno conda y de principio a fin.
☐  Las salidas guardadas corresponden al código final y usan rutas de artefactos coherentes.
☐  Se documentaron versiones de Python y dependencias y se definió la fuente oficial.
☐  El artefacto permite predecir datos nuevos conservando las transformaciones necesarias.
☐  Hay pruebas básicas de carga, preparación, entrenamiento y predicción.

10 · Decisión de negocio, ética y monitoreo
☐  Se definieron capacidad de contacto y costos de falsos positivos/falsos negativos.
☐  El umbral o tamaño de lista priorizada se justifica con costos y capacidad.
☐  Se evaluaron errores del modelo por segmentos relevantes y se discutieron diferencias.
☐  La utilidad de los clusters se validó con perfilado, estabilidad y posible acción; si no aportan, se explica por qué no se usan.
☐  Se describieron límites, población aplicable y riesgos de datos de otra fecha/población.
☐  Se indicó cómo medir éxito en operación, idealmente mediante prueba controlada o seguimiento de campaña.
