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
☑  Datos separados para entrenamiento y prueba; todas las transformaciones se ajustan dentro de los folds de validación cruzada en el flujo canónico.
☑  Predictores iniciales seleccionados y justificados antes del modelamiento.
☑  Relevancia de los 19 predictores originales revisada por permutación en un holdout explicativo de entrenamiento; no se usó para volver a seleccionar el modelo.
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
☑  Uso propuesto y política hipotética explicados con capacidad, umbral OOF, pérdidas y límites de los supuestos; falta aprobación con datos comerciales reales.

07 · Entrega y defensa
☑  README.md incluye problema, objetivos, KPIs, datos, EDA y CRISP-DM.
☑  Los notebooks EDA y modelado se ejecutaron de principio a fin con el entorno conda; el EDA se reejecutó tras cambiar las rutas.
☑  Datasets y carpetas del proyecto están organizados; notebook 02 y comando canónico reproducen la preparación desde el archivo original.
☑  Guion de 9 diapositivas repartido consecutivamente entre 3 integrantes; duración diseñada para 10 minutos.
☐  PPT final generada, revisada contra las fuentes y cronometrada en ensayo grupal.
☐  Cada integrante puede defender cualquier parte del proyecto (pendiente de práctica/verificación).

08 · Validez de la evaluación y fuga de información
☑  El conjunto de prueba final no se usó para elegir modelo ni ajustar hiperparámetros; el artefacto conserva `operating_threshold=None` aunque se haya calculado un umbral OOF hipotético.
☑  La selección de modelos e hiperparámetros se realizó dentro de entrenamiento mediante validación cruzada estratificada.
☑  Imputación, escalado, codificación, PCA/K-Means y balanceo se ajustaron dentro de cada pliegue.
☑  Se definió y reportó un modelo base de comparación.
☑  Se reportó la matriz de confusión en cantidades, además de proporciones y métricas.
☑  Se analizaron precisión-recall y escenarios de umbral en datos OOF; el artefacto conserva operating_threshold=None hasta validar una política real.
☑  Se reportó variabilidad/incertidumbre de las métricas entre pliegues.
☑  Se comparó la sensibilidad con/sin perfiles repetidos usando solo entrenamiento y CV agrupada para evitar que las copias crucen folds; el cambio fue pequeño y no prueba que sean el mismo cliente.
☑  Se documentó que el EDA exploró la cohorte completa y limita la independencia humana del test.
☐  Evaluación confirmatoria en cohorte independiente o temporal no explorada.

09 · Reproducibilidad e implementación
☑  Hay un único flujo oficial desde los datos de entrada hasta el modelo guardado; notebook 02 y script comparten selector, pipeline y persistencia.
☑  El comando descrito en el README ejecuta el flujo canónico y guardó el artefacto.
☑  Los notebooks se ejecutaron desde un kernel limpio del entorno conda y de principio a fin.
☑  Las salidas verificadas se generan bajo `output/graficos/` y `output/modelos/`; las carpetas antiguas están vacías.
☑  Versiones directas fijadas en environment.yml y resolución exacta Windows x64 registrada en environment-win-64.lock.txt.
☑  El artefacto permite puntuar filas de entrada conservando las transformaciones; se verificó mediante carga e inferencia.
☑  Doce pruebas verifican preparación, sensibilidad, ajuste de pipeline, persistencia, carga, inferencia, economía del escenario, umbral y métricas por grupo.

10 · Decisión de negocio, ética y monitoreo
☑  Se calculó una sensibilidad hipotética de capacidad/costos y pérdida usando cargos como proxy; capacidad, margen y eficacia reales siguen pendientes.
☐  El umbral o tamaño de lista priorizada se justifica con costos y capacidad comerciales validados.
☑  Se reportaron métricas de error por género, tercera edad, contrato, servicio y antigüedad, con conteos e intervalos; interpretación descriptiva, no prueba de equidad.
☑  Se decidió no usar K-Means operativamente: separación débil y Recall inferior a predictores originales; queda como análisis/comparador exploratorio.
☑  Se describieron límites de uso, población observada y riesgos de trasladar resultados a otra fecha/población.
☑  Se indicó como siguiente paso medir éxito con datos reales y, preferentemente, una prueba controlada.

Estado de la entrega académica: materiales y guion de defensa preparados; presentación final y ensayos reales todavía deben generarse, revisarse, cronometrarse y registrarse. La validación externa, economía comercial y piloto son pendientes posteriores para cualquier uso operativo.
