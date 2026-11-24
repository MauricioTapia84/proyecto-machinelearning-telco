# Guía de defensa EP3 · Telco Customer Churn

## Mensaje central

El problema es priorizar históricamente a clientes con mayor probabilidad de abandono bajo capacidad limitada. El proyecto compara modelos supervisados, una técnica no supervisada exploratoria y ensambles reproducibles. No afirma retención causal ni despliegue comercial automático.

## Evidencia de optimización

Optuna ajusta `C` del SVM lineal balanceado sobre train. El candidato se ajusta con tres folds en la búsqueda y luego se evalúa con cinco folds estratificados. El test permanece reservado.

## Evidencia de ensambles

Voting soft y Stacking se compararon con el SVM ajustado usando las diez variables seleccionadas.

| Modelo | Recall CV | Recall test | ROC-AUC test |
|---|---:|---:|---:|
| SVM ajustado por Optuna | 80.69% | 82.89% | 85.13% |
| Stacking | 78.06% | 78.61% | 85.23% |
| Voting soft | 67.09% | 68.45% | 85.29% |

El ensamble demuestra comparación y tradeoff: Stacking y Voting tienen mejor o similar ROC-AUC, pero reducen Recall respecto del SVM. La solución canónica conserva el criterio previo de máxima detección.

## Por qué el SVM sigue siendo la solución canónica

No se selecciona por prestigio del algoritmo ni por un único ROC-AUC.

- SVM: Recall CV 80.69%; Recall test 82.89%; Precision test 50.65%; ROC-AUC test 85.13%.
- Stacking: Recall CV 78.06%; Recall test 78.61%; Precision test 52.50%; ROC-AUC test 85.23%.
- Voting: Recall CV 67.09%; Recall test 68.45%; Precision test 59.95%; ROC-AUC test 85.29%.

Stacking mejora ROC-AUC 0.10 puntos porcentuales y Voting 0.16 puntos, pero ambos reducen Recall. El SVM conserva 310 TP frente a 294 TP aproximados de Stacking y detecta más churn bajo el criterio declarado. La adopción de ensamble requeriría aprobar el tradeoff de menor Recall y mayor precisión con capacidad real.

## Prevención de fuga de datos

El split se genera por perfil: 5.617 filas quedan en train y 1.409 en test. Las transformaciones se ajustan con train y los objetos ajustados se usan para transformar test.

El test no interviene en:
- filtro de correlación;
- selección forward SFS;
- comparación de candidatos;
- Optuna;
- ajuste de hiperparámetros;
- selección de ensamble;
- calibración o umbral operativo.

La revisión humana del EDA sobre toda la cohorte es una limitación: no hace que el test sea un experimento totalmente ciego frente a decisiones informadas por el dataset completo.

## Preguntas frecuentes

- ¿Qué es ensamble? Una combinación sistemática de varios clasificadores para reducir varianza o sesgo.
- ¿Por qué el ensamble no reemplaza al SVM? El objetivo declarado prioriza Recall; el ensamble no lo mejora en esta ejecución.
- ¿Qué aporta Optuna? Ajusta `C` sin usar test.
- ¿Cómo se evita fuga? SFS, CV, Optuna y ensambles usan train; test se carga al final.
- ¿Qué métrica decide? Recall es principal y ROC-AUC es criterio complementario.

## Limitaciones

- La probabilidad sigmoid no está calibrada.
- El threshold sigue provisional.
- El test no demuestra generalización temporal ni impacto comercial.
- Los ensambles no prueban que la combinación sea óptima fuera de la cohorte.

## Sesgos y limitaciones

Las métricas por grupo son descriptivas. Existe soporte suficiente en varios segmentos, pero One year y Two year tienen soporte de churn reducido y pueden generar intervalos Wilson amplios. No se afirma injusticia ni causalidad con una sola partición post hoc.

La mitigación requiere: soporte mínimo, diferencia estable, impacto operativo medible, reproducibilidad temporal y revisión humana. No se permite excluir automáticamente clientes por grupo ni presentar una diferencia histórica como prueba de discriminación.

Otras limitaciones: el score no es una probabilidad calibrada, el umbral es provisional, el test es interno y no representa necesariamente una cohorte futura, y el modelo no mide retención efectiva.

## Criterio Recall y costo operativo

Recall se priorizó porque el objetivo consiste en detectar la mayor cantidad posible de abandonos reales bajo una revisión humana limitada. El cálculo es TP/(TP+FN): mide qué proporción de los 374 churn reales del test fueron detectados.

En el test vigente el SVM detectó 310 de 374, Recall 82.89%. El costo de esa decisión fue 302 falsos positivos. Por ello no se afirma que Recall maximiza beneficio económico: solo maximiza detección y debe validarse contra capacidad y costo real.

## Desempeño histórico versus impacto comercial

El resultado histórico significa que el modelo ordenó 374 abandonos reales del test y detectó 310 bajo el punto operativo por defecto. Esto permite evaluar clasificación, pero no demuestra que una campaña conserve clientes ni que genere ahorro.

El impacto comercial exige: costes reales de contacto, valor esperado del cliente, capacidad del equipo, tasa real de retención, diseño controlado y observación posterior. Un piloto debe comparar tratamiento y control y reportar intervalos de confianza y costo neto.
