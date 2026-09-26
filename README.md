# Penguins Species Classifier

This repository is a hands-on example of turning a notebook-based ML workflow into a
clean, maintainable, production-style Python project. The same Palmer Penguins
species classifier is built across multiple stages, with each stage improving the
structure, reproducibility, and handoff quality of the code.

## Project stages

1. `stage0_prototype/`: the original messy notebook prototype
2. `stage1_structured_code/`: the same workflow reorganized into clearer code
3. `stage2_functions/`: logic split into reusable functions
4. `stage_3_modularized/`: a modular project with a dedicated training workflow and
   prediction CLI

The early stages focus on learning how the workflow evolves from ad hoc notebook
code to a more robust engineering structure. The final stage represents the
production-ready version of the project.

## Stage 3: modularized ML project

Stage 3 turns the pipeline into a modular Python project with separate training and
prediction workflows.

### Train a model

```bash
conda activate data-karigor
python train.py
```

From the `stage_3_modularized/` directory, this reads the Penguins dataset, cleans it,
fits the configured models, selects the best performer, and saves trained artifacts in
`artifacts/`:

- `model.joblib`
- `scaler.joblib`
- `encoders.joblib`
- `feature_cols.json`
- `metrics.json`

### Predict a species

```bash
python predict.py
```

The prediction script loads the saved artifacts and classifies one raw penguin record
from `configs/sample.json`.

### Test the project

```bash
python -m pytest
```

This project is intended to demonstrate how ML work can move from exploratory notebooks
to modular, testable, and more production-friendly code.
