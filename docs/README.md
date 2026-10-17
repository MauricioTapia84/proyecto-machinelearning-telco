# Documentación del Proyecto

Este directorio centraliza la documentación técnica del proyecto.

## Objetivo

Registrar la estructura, el flujo de trabajo y la trazabilidad del análisis de churn de clientes.

## Flujo recomendado

1. Preparación y EDA en el notebook 1; sus artefactos quedan bajo `output/`.
2. Modelado y evaluación en el notebook 2, con selección y persistencia compartidas con los scripts.
3. Ejecución reproducible desde la raíz mediante `python scripts/run_project.py`.
4. Inferencia de registros nuevos con el pipeline persistido en `output/modelos/`.
5. Revisión hipotética de pérdidas y métricas por grupo mediante scripts específicos; los resultados quedan en `output/resultados/`.

## Archivos importantes

- `README.md`: descripción general del proyecto.
- `notebooks/01_EDA_y_Limpieza.ipynb`: EDA, limpieza y tratamiento de nulos.
- `notebooks/02_Modelado_y_Evaluacion.ipynb`: modelado y evaluación.
- `src/pipeline.py`: preprocesamiento y pipeline reutilizable.
- `src/modeling.py`: selección con CV y persistencia del pipeline.
- `src/inference.py`: carga del artefacto y predicción para registros nuevos.
- `src/paths.py`: rutas canónicas de salidas.
- `src/decision.py`: selección de umbral bajo capacidad y costos hipotéticos.
- `src/group_metrics.py`: conteos, tasas e intervalos por grupo.
- `src/sensitivity.py`: comparación de políticas de deduplicación solo en entrenamiento.
- `src/train.py`: entrenamiento del modelo y guardado.
- `scripts/generate_split.py`: generación de la partición estratificada train/test.
- `scripts/run_project.py`: única entrada oficial para selección, entrenamiento, evaluación final y persistencia del modelo.
- `scripts/evaluate_operating_scenarios.py`: sensibilidad hipotética a capacidad, eficacia y costo de contacto.
- `scripts/evaluate_group_metrics.py`: errores por grupos en el test reservado.
- `docs/estructura_proyecto.md`: standard de estructura profesional recomendado.
- `environment.yml`: dependencias directas con versiones fijadas.
- `environment-win-64.lock.txt`: resolución exacta Conda para Windows x64.
- `docs/guia_defensa_ep2.md`: preguntas, respuestas y pauta de ensayo individual.
- `docs/presentacion_ep2.md`: guion consecutivo de 10 minutos para tres integrantes, notas por diapositiva, prompt para generar PPT con IA, checklist de revisión y banco de 20 preguntas con respuestas modelo.
- `docs/checklist.md`: estado de avance de la rúbrica, con pendientes académicos separados de los requisitos para uso operativo.

Los scripts ejecutables se mantienen en la raíz del repositorio, dentro de `scripts/`; el código reutilizable y la orquestación del entrenamiento están en `src/`. Para entrenar, se documenta y utiliza exclusivamente `python scripts/run_project.py`; se retiraron las entradas duplicadas `main.py` y `scripts/train_model.py`.

La política económica es un escenario hipotético basado en cargos brutos del dataset, no una valoración financiera observada. K-Means queda como análisis exploratorio y no define acciones comerciales.
