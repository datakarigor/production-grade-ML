from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/penguins.csv"
FEATURE_COLS = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "bill_ratio",
    "body_mass_scaled",
    "island",
    "sex",
]
TARGET_COL = "species"

RANDOM_SEED = 42
TEST_SIZE = 0.2

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = "artifacts/model.joblib"
SCALER_PATH = "artifacts/scaler.joblib"
ENCODERS_PATH = "artifacts/encoders.joblib"
FEATURE_COLS_PATH = "artifacts/feature_cols.json"
METRICS_PATH = "artifacts/metrics.json"
SAMPLE_PATH = "configs/sample.json"

MODEL_CONFIG = [
    (
        "Random Forest",
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": RANDOM_SEED},
    ),
    (
        "Logistic Regression",
        LogisticRegression,
        {"max_iter": 10000, "random_state": RANDOM_SEED},
    ),
    (
        "Gradient Boosting",
        GradientBoostingClassifier,
        {"random_state": RANDOM_SEED},
    ),
]
