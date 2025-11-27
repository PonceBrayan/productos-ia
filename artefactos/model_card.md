# Model Card — Riesgo de Deserción (MLP + OHE)

**Fecha exportación:** 2025-11-09T09:39:05.872509
**Artefacto:** pipeline_mlp_ohe.joblib
**Features (7):** PromedioPonderado, DeudaFinanciera, SintomasDepresion, Edad, SituacionLaboral, Trasladado, Asistencia
**Clases (orden):** ['Bajo', 'Medio', 'Alta']
**Dimensión de entrada efectiva (one-hot):** 23

## Datos y Preprocesamiento
- Entradas categóricas con dominios cerrados (ver `input_schema.json`).
- One-Hot Encoding con categorías fijas; `handle_unknown="ignore"`.

## Modelo
- `sklearn.neural_network.MLPClassifier` dentro de `Pipeline` con preprocesamiento.

## Versionado
- Python: 3.12.2
- scikit-learn: 1.5.1
- numpy: 1.26.4
- SO: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.36
