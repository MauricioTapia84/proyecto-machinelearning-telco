# Documentación del Proyecto

Este directorio centraliza la documentación técnica del proyecto.

## Objetivo

Registrar la estructura, el flujo de trabajo y la trazabilidad del análisis de churn de clientes.

## Flujo recomendado

1. ETL y EDA en el notebook 1.
2. División estricta de train/test antes del análisis profundo.
3. Modelado y evaluación en el notebook 2.
4. Persistencia del modelo y exportación de resultados.

## Archivos importantes

- `README.md`: descripción general del proyecto.
- `notebooks/01_EDA_y_Limpieza.ipynb`: EDA, limpieza y tratamiento de nulos.
- `notebooks/02_Modelado_y_Evaluacion.ipynb`: modelado y evaluación.
- `src/pipeline.py`: preprocesamiento y pipeline reutilizable.
- `src/train.py`: entrenamiento del modelo y guardado.
- `scripts/generate_split.py`: generación de la partición estratificada train/test.
- `scripts/train_model.py`: ejecución del entrenamiento del modelo.
- `scripts/run_project.py`: flujo completo del proyecto.
- `docs/estructura_proyecto.md`: standard de estructura profesional recomendado.
