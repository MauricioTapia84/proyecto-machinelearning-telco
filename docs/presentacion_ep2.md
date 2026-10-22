# Guion de presentación EP2 · Telco Customer Churn

## Mensaje central

El proyecto no intenta “adivinar” el futuro ni automatizar una campaña. Construye un score para ordenar clientes con mayor riesgo histórico de churn y ayudar a priorizar revisiones humanas. La evidencia debe avanzar desde el problema, pasar por la calidad y la evaluación, y terminar en una decisión prudente.

Duración sugerida: 10 minutos · 3 integrantes · 9 diapositivas consecutivas.

## Diapositiva 1 · Problema y decisión

**En pantalla**

- Telco Customer Churn.
- Pregunta: ¿a quién conviene revisar primero?
- Salida: score de riesgo, no oferta automática.
- Prioridad: Recall; ROC-AUC como apoyo.

**Qué decir**

“Partimos de una decisión de negocio concreta: un equipo de retención tiene capacidad limitada y necesita priorizar revisiones. El modelo ordena clientes según un score de riesgo histórico. No interpretamos el score como probabilidad calibrada ni como evidencia de que una campaña retendrá clientes.”

## Diapositiva 2 · Datos y calidad

**En pantalla**

- 7.043 registros · 19 predictores originales · `Churn` binaria.
- 1.869 churn: 26,54%.
- `TotalCharges` llegó como texto; 11 espacios corresponden a `tenure = 0`.
- 22 perfiles repetidos al ignorar `customerID`; sensibilidad solo en train.

**Qué decir**

“Antes de modelar verificamos qué representa una fila y qué problemas tiene el archivo. Convertimos `TotalCharges` a numérico, tratamos los casos sin antigüedad y excluimos el identificador. Los perfiles repetidos no se interpretan automáticamente como la misma persona; se controlan para no inflar la validación.”

## Diapositiva 3 · EDA: qué aprendimos

**En pantalla**

- Más churn descriptivo en contratos mes a mes y clientes con menor antigüedad.
- Correlación `tenure`–`TotalCharges`: 0,826.
- Boxplots y tasas por categoría orientan hipótesis.
- Asociación no es causalidad.

**Qué decir**

“El EDA muestra patrones plausibles para orientar el modelo. `TotalCharges` tiene cola derecha y no se escala directamente sin corregirla. Los boxplots muestran diferencias de distribución por clase, pero no son una prueba causal ni una regla automática de selección. Como exploramos la cohorte completa, reconocemos que el test no es totalmente ciego frente a decisiones humanas.”

## Diapositiva 4 · Evaluación sin fuga

**En pantalla**

```text
CSV crudo → split por perfil → train/test transformados
                              ↓
                 selección y ajuste solo con train
                              ↓
                    test reservado para el final
```

- Train: 5.617 filas · Test: 1.408 filas.
- El modelado consume `data/processed/`.
- Sin SMOTE; `class_weight='balanced'` en SVM.

**Qué decir**

“El CSV crudo solo genera las matrices procesadas. Las transformaciones se ajustan con train; test se transforma con esos objetos, pero no participa en SFS, comparación ni Optuna. La clase minoritaria se pondera en el SVM; no fabricamos ejemplos sintéticos y no modificamos test.”

## Diapositiva 5 · Reducción de variables

**En pantalla**

- 46 columnas transformadas iniciales.
- Filtro `|correlación| > 0,95`: elimina 13 redundantes.
- `SequentialFeatureSelector`, forward, 5 folds, 10 columnas.
- `LinearRegression` se usa solo como screening con `neg_mean_squared_error`.
- Boxplots: diagnóstico descriptivo, no selección automática.

**Qué decir**

“Primero quitamos redundancias evidentes en train. Después SFS agrega variables de forma incremental y evalúa cada conjunto con validación cruzada. Usamos `LinearRegression` como estimador de screening porque se solicitó una regresión lineal simple; no lo presentamos como el clasificador de churn. La decisión final continúa siendo supervisada y binaria.”

## Diapositiva 6 · Comparación y elección

**En pantalla**

- DummyClassifier como referencia.
- Regresión Logística, Random Forest, Gradient Boosting y SVM.
- Criterio principal: Recall; desempate: ROC-AUC.
- Ganador: SVM lineal balanceado, Optuna, `C≈0,1683`.
- PCA/K-Means: comparación exploratoria, no política comercial.

**Qué decir**

“No elegimos por complejidad ni por prestigio del algoritmo. Comparamos candidatos con el mismo protocolo de entrenamiento. Como el objetivo es detectar abandono, priorizamos Recall. SVM ofrece una buena relación entre detección y ordenamiento, y el peso de clase maneja el desbalance sin SMOTE.”

## Diapositiva 7 · Resultado en test

**En pantalla**

| Modelo | Variables | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM completo | 46 | 0,8430 | 0,4693 | 0,6030 | 0,8474 |
| SVM reducido | 10 | 0,8430 | 0,4693 | 0,6030 | 0,8445 |

- Matriz reducida: `[[699, 346], [57, 306]]`.
- 306 de 363 churn detectados.
- Pérdida de ROC-AUC: 0,0028.

**Qué decir**

“Reducir de 46 a 10 variables conserva Accuracy, Recall, Precision y F1 en el test, con una pérdida pequeña de ROC-AUC. Por eso la simplificación es defendible: reduce complejidad sin sacrificar la decisión al umbral usado. Aun así, 346 falsos positivos muestran que mejorar Recall tiene un costo operativo.”

## Diapositiva 8 · Interpretación y límites

**En pantalla**

- `decision_function` no es probabilidad calibrada.
- Priorización no equivale a retención efectiva.
- Auditorías por grupo son descriptivas, no certificación de equidad.
- Falta cohorte temporal/externa y prueba controlada.

**Qué decir**

“El resultado permite ordenar una cartera histórica, no prometer que los clientes serán retenidos. Antes de desplegar habría que validar en una cohorte nueva, revisar estabilidad por grupos, acordar capacidad y costos, calibrar si se necesitan probabilidades y medir el efecto de una campaña con control.”

## Diapositiva 9 · Cierre

**En pantalla**

1. El problema se convirtió en priorización supervisada.
2. La evaluación mantiene el test reservado.
3. La reducción deja 10 variables sin pérdida relevante en las métricas de clasificación.
4. El siguiente paso es validación externa y experimento de impacto.

**Qué decir**

“La conclusión no es que el modelo resuelva el churn. La conclusión es más precisa: con la evidencia disponible, un SVM lineal balanceado puede priorizar clientes históricos y una representación de 10 variables mantiene el desempeño principal. La decisión comercial requiere nueva evidencia y medición causal.”

## Prompt para generar las diapositivas

Crear una presentación académica de ciencia de datos, sobria y legible, con 9 diapositivas y duración total de 10 minutos. Usar una narrativa progresiva: problema de negocio → datos y calidad → EDA → evaluación sin fuga → selección de variables → comparación → test → límites → cierre. Mantener la paleta azul petróleo, gris y un acento naranja para `Churn`. Incluir un diagrama simple del flujo train/test, una figura de boxplots, una tabla corta de resultados y la matriz de confusión. No inventar cifras ni usar SMOTE como parte del flujo final. Todas las cifras deben coincidir con este guion y con `output/resultados/comparacion_variables_reducidas.csv`.

## Banco breve de preguntas

1. **¿Por qué Recall?** Porque el objetivo exploratorio es detectar la mayor proporción posible de abandonos; el costo es un aumento de falsos positivos.
2. **¿Por qué no Accuracy?** Con 26,54% de churn puede ocultar un mal desempeño sobre la clase minoritaria.
3. **¿Por qué Yeo-Johnson para `TotalCharges`?** Su sesgo derecho es alto y Yeo-Johnson reduce la cola sin exigir positividad estricta.
4. **¿Por qué no codificar categorías como 0, 1, 2?** Crearía un orden artificial; One-Hot Encoding respeta su naturaleza nominal.
5. **¿Por qué `LinearRegression`?** Solo como estimador de screening para SFS; el clasificador final es SVM.
6. **¿Por qué no SMOTE?** El balanceo por pesos fue suficiente y evita observaciones sintéticas en una matriz con One-Hot.
7. **¿Se usó test en SFS?** No; SFS, Optuna y la comparación usan train.
8. **¿Qué significa 0,8430 de Recall?** Se detectaron 306 de 363 abandonos reales del test.
9. **¿Qué significa la precisión 0,4693?** Aproximadamente 47% de los marcados como positivos fueron churn en este punto de operación.
10. **¿El score es probabilidad?** No; es la salida de decisión del SVM y no está calibrada.
11. **¿Qué aporta K-Means?** Una comparación no supervisada; no es el mecanismo de predicción ni de contacto.
12. **¿Cuál es el siguiente paso?** Cohorte nueva o temporal, validación de estabilidad y piloto controlado de retención.
