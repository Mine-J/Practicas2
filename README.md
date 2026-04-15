# Sistema de Prediccion de Acciones con XGBoost

Proyecto de analisis y prediccion de tendencia de acciones (sube/baja) usando indicadores tecnicos y un modelo `XGBoost`.

El script principal es `Memoria_II.py` y funciona con menu interactivo en consola.

## Objetivo

Entrenar y usar modelos de clasificacion para estimar si el precio de una accion subira o bajara en un horizonte temporal definido (1, 3, 5, 7, 10, 15 o 20 dias).

## Estructura del proyecto

- `Memoria_II.py`: script principal con carga de datos, ingenieria de variables, entrenamiento y predicciones.
- `requirements.txt`: dependencias de Python.
- `individual_stocks_5yr/`: dataset de entrada en CSV (un archivo por ticker).
- `Modelos/`: carpeta de salida para modelos entrenados (se crea/usa automaticamente).
- `recomendaciones_7dias.csv`: ejemplo de salida con recomendaciones globales.

## Requisitos

- Python 3.10 o superior (recomendado)
- Dependencias de `requirements.txt`

Instalacion:

```bash
pip install -r requirements.txt
```

## Formato de datos esperado

Cada CSV en `individual_stocks_5yr/` debe incluir estas columnas:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `Name` (ticker)

El script filtra automaticamente fechas entre `2013-01-01` y `2020-12-31`.

## Ejecucion

Desde la raiz del proyecto:

```bash
python Memoria_II.py
```

Al iniciar, el programa carga y limpia datos, y luego muestra un menu con 5 opciones.

## Opciones del menu

1. Comparar horizontes temporales: evalua rendimiento para varios horizontes y muestra una recomendacion.
2. Entrenar modelo especifico: entrena un modelo para el horizonte elegido y guarda artefactos.
3. Predecir accion especifica: usa un modelo entrenado para analizar un ticker concreto.
4. Analisis general: analiza todas las acciones y genera top de compras/ventas.
5. Salir.

## Artefactos generados

Al entrenar (opcion 2), se guardan en `Modelos/modelo_<N>_dias/`:

- `modelo_<N>_dias.pkl`
- `features_<N>_dias.pkl`
- `metricas_<N>_dias.pkl`

Al ejecutar analisis general (opcion 4), se genera:

- `recomendaciones_<N>dias.csv`

## Notas

- Para usar opciones 3 y 4, primero debes haber entrenado al menos un modelo (opcion 2).
- El enfoque es educativo/academico y no constituye asesoramiento financiero.
