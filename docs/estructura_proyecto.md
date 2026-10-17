# Estructura del proyecto

La organización separa interfaces ejecutables, código reutilizable, análisis interactivo, datos y resultados. `scripts/` y `src/` permanecen en la raíz: los scripts son puntos de entrada para personas y automatizaciones; `src/` contiene la implementación que esos puntos de entrada reutilizan.

```text
proyecto/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
│   ├── 01_EDA_y_Limpieza.ipynb
│   └── 02_Modelado_y_Evaluacion.ipynb
├── output/
│   ├── graficos/
│   ├── modelos/
│   └── resultados/
├── scripts/
│   ├── run_project.py
│   ├── generate_split.py
│   ├── evaluate_operating_scenarios.py
│   ├── evaluate_group_metrics.py
│   └── calculate_business_kpis.py
├── src/
│   ├── data.py
│   ├── decision.py
│   ├── group_metrics.py
│   ├── inference.py
│   ├── modeling.py
│   ├── paths.py
│   ├── pipeline.py
│   ├── sensitivity.py
│   ├── split_data.py
│   └── train.py
├── environment.yml
├── environment-win-64.lock.txt
├── README.md
└── .gitignore
```

`scripts/run_project.py` es la **única entrada oficial** para seleccionar, entrenar, evaluar y persistir el modelo. El archivo delega el flujo a `src/train.py`, que reutiliza la selección de `src/modeling.py` y el pipeline de `src/pipeline.py`. Las entradas duplicadas `main.py` y `scripts/train_model.py` se retiraron para que la ejecución documentada no sea ambigua.

## Responsabilidad por carpeta

- **`src/`**: funciones, clases y orquestación que pueden reutilizar notebooks, scripts y pruebas.
- **`scripts/`**: comandos ejecutables enfocados; se mantienen en la raíz y llaman a la lógica de `src/`, evitando duplicarla.
- **`notebooks/`**: exploración, narrativa y evaluación legible. El notebook 02 comparte selector, pipeline y guardado con el flujo canónico; el notebook 01 produce artefactos auxiliares de EDA.
- **`data/raw/`**: fuente original. **`data/processed/`**: exportaciones auxiliares que pueden regenerarse; no son la entrada del entrenamiento canónico.
- **`output/`**: artefactos generados; gráficos en `graficos/`, modelos en `modelos/` y reportes tabulares en `resultados/`.
- **`docs/`**: metodología, estructura, checklist y materiales de defensa.

## Flujo reproducible de entrenamiento

Desde la raíz, activar el entorno Conda y ejecutar:

```powershell
conda activate telco-ml
python scripts\run_project.py
```

El flujo lee el archivo fuente, prepara la cohorte, crea el split estratificado, selecciona con validación cruzada solo en entrenamiento, ajusta el pipeline final, reporta el test y persiste el artefacto completo en `output/modelos/`. El notebook 02 llama a los mismos componentes reutilizables para su análisis; no se mantiene un segundo script de entrenamiento.

Los demás comandos de `scripts/` son tareas auxiliares con propósito separado (exportación de particiones, KPIs o auditorías); no son alternativas al flujo canónico de entrenamiento.

## Separación de responsabilidades metodológicas

### Notebook 1: EDA y limpieza

Describe calidad, distribuciones, patrones y correlaciones. Puede exportar matrices y transformadores exploratorios bajo `output/`, pero estos no alimentan el entrenamiento canónico.

### Notebook 2: modelado y evaluación

Lee nuevamente la fuente original y utiliza `src/pipeline.py` y `src/modeling.py`. El pipeline ajusta las transformaciones dentro de cada fold de CV, compara candidatos, selecciona el modelo, reporta el test al final y guarda el pipeline de inferencia.

## Entorno

`environment.yml` declara las dependencias directas fijadas. `environment-win-64.lock.txt` registra la resolución explícita para Windows x64. La configuración se detalla en el [README](../README.md).
