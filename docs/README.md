# Documentación del proyecto

Este directorio centraliza la metodología, la revisión de coherencia y los materiales de defensa del proyecto Telco Customer Churn.

## Flujo recomendado

1. Ejecutar el EDA en [01_EDA_y_Limpieza.ipynb](../notebooks/01_EDA_y_Limpieza.ipynb).
2. Generar las matrices procesadas con `python scripts\generate_split.py`.
3. Evaluar la reducción de variables con `python scripts\evaluate_reduced_features.py`.
4. Entrenar, evaluar al final y persistir con `python scripts\run_project.py`.
5. Revisar [02_Modelado_y_Evaluacion.ipynb](../notebooks/02_Modelado_y_Evaluacion.ipynb) y los artefactos de [output/](../output/).

El flujo canónico parte de `data/processed/`, usa solo train para selección y ajuste, y reserva test para la evaluación final. No utiliza SMOTE: el SVM final usa `class_weight='balanced'`.

## Archivos principales

- [README.md](../README.md): objetivo, metodología, transformaciones y resultados.
- [guia_defensa_ep2.md](./guia_defensa_ep2.md): explicación técnica y preguntas de defensa.
- [presentacion_ep2.md](./presentacion_ep2.md): guion de nueve diapositivas con storytelling.
- [revision_integral_ep2.md](./revision_integral_ep2.md): matriz de coherencia entre etapas.
- [checklist.md](./checklist.md): revisión de requisitos y pendientes.
- [estructura_proyecto.md](./estructura_proyecto.md): responsabilidades y flujo de carpetas.
- [processed_data.py](../src/processed_data.py): creación de train/test procesados.
- [modeling.py](../src/modeling.py): filtro de correlación, SFS y selección de modelos.
- [train.py](../src/train.py): entrenamiento sin cargar test hasta el final.
- [evaluate_reduced_features.py](../scripts/evaluate_reduced_features.py): auditoría de reducción y comparación completa/reducida.
- [run_project.py](../scripts/run_project.py): entrenamiento oficial y persistencia del artefacto.

## Mensaje metodológico

La decisión de transformación depende de la distribución: Yeo-Johnson para `TotalCharges`, `StandardScaler` para numéricas que necesitan escala comparable y One-Hot Encoding para categorías nominales. La selección deja 10 columnas transformadas mediante filtro de correlación y `SequentialFeatureSelector`; `LinearRegression` se utiliza únicamente como screening, no como clasificador.

La conclusión vigente es que el SVM lineal balanceado conserva Recall, Precision, F1 y Accuracy del modelo completo en el test actual, con una pérdida de 0,0028 en ROC-AUC. Esto respalda simplificar la representación, pero no prueba causalidad, rentabilidad ni validez fuera de la cohorte observada.
