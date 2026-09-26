# Penguins Species Classifier

Stage 3 turns the Stage 2 notebook pipeline into a modular Python project with a
separate training workflow and prediction CLI.

## Train a model

```bash
conda activate data-karigor
python train.py
```

This reads the penguins dataset, cleans it, fits the configured models, selects the
best performer, and saves the trained artifacts in `artifacts/`:

- `model.joblib`
- `scaler.joblib`
- `encoders.joblib`
- `feature_cols.json`
- `metrics.json`

## Predict a species

```bash
python predict.py
```

The prediction script loads the saved artifacts and classifies one raw penguin
record from `configs/sample.json`.

## Test the project

```bash
python -m pytest
```