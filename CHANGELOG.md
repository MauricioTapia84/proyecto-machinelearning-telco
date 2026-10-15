# Changelog

## Commits

### 15 de octubre de 2026

- #### 2.8. Definición de KPIs y métricas de negocio

  - Se documentó el problema de negocio como una priorización de clientes con riesgo de churn para apoyar acciones de retención.
  - Se incorporó una tabla que diferencia indicadores de negocio de métricas técnicas.
  - Se calculó la tasa de churn del dataset completo: 1.869 de 7.043 clientes, equivalente a 26,54%.
  - Se documentó el Recall del SVM seleccionado: 77,15% en el conjunto de prueba.
  - Se dejó explícitamente pendiente el cálculo puntual de captura de churn en el 20% de mayor riesgo, porque el notebook actual solo persiste la curva de ganancia y no ese corte numérico.
  - No se inventaron metas, costos, ingresos protegidos ni tasas futuras de retención.

- #### 2.9. Cálculo de captura de churn en clientes de mayor riesgo

  - Se calculó la captura de churn en el 20% de mayor riesgo: 191 de 372 abandonos del conjunto de prueba, equivalente a 51,34%.
  - Se verificó que el Recall recalculado del SVM coincide con el resultado documentado: 77,15%.
  - Se agregó [scripts/calculate_business_kpis.py](scripts/calculate_business_kpis.py) para reproducir los tres indicadores definidos.
  - Se completó el primer subpunto del checklist al contar con problema, objetivo y KPIs respaldados por valores calculables.