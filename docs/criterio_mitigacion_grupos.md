# Criterio de mitigación por grupos

La mitigación no se decide por una diferencia aislada del test. Debe demostrar simultáneamente:

1. **Soporte suficiente:** n >= 30, churn >= 10 y permanencia >= 10, salvo aprobación metodológica documentada.
2. **Diferencia estable:** el intervalo Wilson 95% no debe solaparse o la diferencia debe ser material frente al intervalo operativo.
3. **Relevancia operativa:** la diferencia debe afectar Recall, precisión o falsos positivos con costo estimable.
4. **Reproducibilidad:** la señal debe aparecer en más de una partición temporal o externa cuando exista información disponible.
5. **Revisión humana:** negocio, privacidad y equidad deben aprobar el objetivo y los límites de mitigación.

## Acciones permitidas

- Ajustar umbral global solo si mejora Recall/precision sin exceder capacidad.
- Usar umbrales locales solo con validación externa y sin producir una decisión de exclusión automática.
- Reentrenar o ampliar datos representativos si falta soporte.
- Mantener supervisión humana si la diferencia no es estable.

## Acciones no permitidas

- Usar una variable sensible como criterio exclusivo de aprobación o rechazo.
- Usar exclusión automática por grupo.
- Ajustar el modelo para forzar métricas iguales sin aumentar evidencia o validar impacto.
- Presentar una diferencia post hoc como injusticia probada o causal.

## Evidencia mínima

Las métricas por grupo deben incluir n, apoyo de cada clase, Recall, precisión, falsos positivos, intervalos Wilson y advertencia de soporte.
