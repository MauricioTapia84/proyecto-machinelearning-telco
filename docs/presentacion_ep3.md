# Guion de presentación EP3 · Telco Customer Churn

Duración: 10 minutos · presentación individual/grupal según pauta.

## 1. Problema y objetivo

Ordenar clientes con riesgo histórico de abandono bajo capacidad limitada.

**KPI principal: Recall** — TP/(TP+FN). Responde cuántos clientes que realmente abandonan fueron detectados. El modelo encontró 310 de 374 churn históricos: Recall 82.89%. El costo son 302 falsos positivos. ROC-AUC apoya la lectura de ordenamiento, pero no reemplaza el objetivo operativo.

El producto no automatiza campañas ni demuestra retención.

## 2. Datos y calidad

Dataset público de 7.043 registros, 19 variables originales y 26.54% de churn. `TotalCharges`, duplicados de perfil, identificador y transformaciones se explican con trazabilidad.

## 3. Metodología y prevención de fuga

CRISP-DM, **split por perfil, transformaciones ajustadas solo con train, SFS, CV, Optuna y ensambles sobre train. Test reservado se carga después del ajuste final. Reconocer que el EDA completo limita la ceguera absoluta frente a decisiones humanas.** Sin SMOTE; pesos balanceados en SVM.

## 4. Modelos supervisados

Dummy, Regresión Logística, Random Forest, Gradient Boosting y SVM. Comparación con validación cruzada estratificada.

## 5. Optimización

Optuna ajusta `C` del SVM lineal sobre train. Muestra reproducible de parámetros y métricas; test no participa.

## 6. Técnica de ensamble

Voting soft y Stacking se compararon con el SVM ajustado:

- SVM: Recall CV 80.69%, test 82.89%, ROC-AUC test 85.13%.
- Stacking: Recall CV 78.06%, test 78.61%, ROC-AUC test 85.23%.
- Voting: Recall CV 67.09%, test 68.45%, ROC-AUC test 85.29%.

**Por qué no reemplaza al SVM:** Stacking mejora ligeramente ROC-AUC test de 85.13% a 85.23%, pero reduce Recall test de 82.89% a 78.61%. Voting mejora ROC-AUC a 85.29%, pero reduce Recall a 68.45%. La adopción requeriría aprobar explícitamente el tradeoff entre detección y alertas.

Conclusión: la combinación mejora poco el ROC-AUC, pero reduce Recall. Se conserva el SVM por el objetivo declarado de detección.

## 7. Evaluación final

Test: 1.409 clientes, 374 churn. Matriz: TN 733, FP 302, FN 64, TP 310. Accuracy 74.02%, Precision 50.65%, Recall 82.89%, F1 62.88%, ROC-AUC 85.13%.

## 8. Sesgos y límites

El score no es probabilidad calibrada. Analizar por grupos, soporte insuficiente, límites Wilson y cautela sobre causalidad. Falta validación temporal y piloto controlado para aprobación comercial.

**Sesgos y límites:** métricas por grupo son post hoc y requieren soporte suficiente e intervalos Wilson; no prueban injusticia ni causalidad. Grupos pequeños pueden generar intervalos amplios.

**Desempeño histórico vs impacto real:** 310 TP describen clasificación sobre una partición histórica. No prueban retención. El impacto requiere costos reales, capacidad, piloto controlado y observación posterior.

## 9. Conclusiones

Se logró optimización, comparación supervisada, ensamble, validación cruzada y solución justificable. El siguiente paso externo es aprobación temporal, calibración y piloto de retención.
