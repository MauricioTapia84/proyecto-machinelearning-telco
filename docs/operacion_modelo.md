# Operación del modelo de churn

## Regla de reentrenamiento

Reentrenar únicamente con una ejecución reproducible usando `python scripts/run_project.py` y el entorno `telco-ml`.

Activar reentrenamiento cuando:
- PSI supera 0.20 en variables relevantes durante dos periodos consecutivos;
- Recall cae fuera del intervalo Wilson/temporal aprobado;
- el umbral no conserva la capacidad operativa acordada;
- la distribución de resultados cambia después de una actualización de datos o producto;
- se añade una cohorte temporal con al menos soporte mínimo por grupo.

El test nunca participa en selección, umbral, calibración ni decisión de reentrenamiento.

## Validación previa a adopción

Antes de publicar un nuevo artefacto:
1. Ejecutar pruebas y script canónico.
2. Regenerar SHA-256 y verificar artefacto v2.
3. Comparar métricas y umbral contra el artefacto aprobado.
4. Ejecutar Docker Compose y `/health`/`/predict`.
5. Aprobar explícitamente el cambio con dueño técnico y de negocio.

## Rollback

Mantener el artefacto previo y su sidecar SHA-256 fuera de `output/modelos/` en una ubicación versionada o copia inmediatamente anterior.

Pasos:
1. Detener el servicio o desplegar el contenedor anterior.
2. Restaurar el `.joblib` aprobado junto con su `.sha256.txt`.
3. Ejecutar `python scripts/smoke_test_api.py`.
4. Verificar `artifact_schema`, `artifact_sha256` y `/health`.
5. Registrar fecha, motivo y responsable.

No realizar rollback automático basándose solo en una caída corta o ruido estadístico.

## Piloto controlado

No puede medirse retención real sin la intervención comercial. Requiere:
- Grupo aleatorio: contactos con intervención estándar;
- Control: intervención actual o sin contacto, según diseño ético aprobado;
- Identificador anónimo persistente;
- Fecha de exposición y fecha observada de abandono;
- Costo por contacto;
- Valor o ahorro por retención efectiva;
- Capacidad real del equipo.

Medir:
- Tasa observada de churn por grupo;
- Diferencia de retención con intervalo de confianza;
- Precision y recall en la cohorte pilotada;
- Costo esperado por cliente contactado;
- Net savings frente al control.

El piloto debe aprobarse por revisión ética/legal/comercial antes de ejecutarse.

## Estado de aprobación formal

Los siguientes estados están definidos:
- `provisional`: estado actual; no autoriza campañas ni cambio comercial automático.
- `approved`: requiere piloto aprobado, capacidad real, Recall/precision aceptados y decisión de calibración registrada.

El script `evaluate_controlled_pilot.py` genera evidencia económica, pero no aprueba automáticamente el umbral.
El CSV sintético no constituye evidencia comercial ni reemplaza datos reales aprobados.

## Decisión de calibración

Sin validación temporal externa, el modelo conserva sigmoid como aproximación y `probability_calibrated=false`.
Isotónica solo puede adoptarse si existe una comparación externa registrada con estabilidad de Brier y tradeoff Recall/Precision.

## Comandos operativos

- Backup: `python scripts/backup_model_artifact.py --label approved`
- Rollback: `python scripts/rollback_model_artifact.py --backup <ruta>/telco_churn_pipeline.joblib`
- Drift: `python scripts/evaluate_drift.py`
- Decisión reentrenamiento: `python scripts/evaluate_retraining_alert.py`
- Smoke API: `python scripts/smoke_test_api.py --url http://127.0.0.1:8000`
- Carga: `python scripts/load_test_api.py --url http://127.0.0.1:8000 --requests 20 --workers 4`

Los scripts de backup y rollback verifican SHA-256 antes y después de restaurar.
