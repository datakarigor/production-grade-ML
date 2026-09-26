from sklearn.metrics import accuracy_score

from configs.config import MODEL_CONFIG, TARGET_COL
from src.data import clean_data, load_data
from src.evaluate import evaluate, evaluate_models
from src.features import build_features
from src.model import train_all_models


def _train_and_test_data():
    df = clean_data(load_data("data/penguins.csv"))
    features, _ = build_features(df)
    target = df[TARGET_COL]

    train_size = len(df) - 10
    features_train = features.iloc[:train_size]
    target_train = target.iloc[:train_size]
    features_test = features.iloc[train_size:]
    target_test = target.iloc[train_size:]

    return features_train, target_train, features_test, target_test


def test_evaluate_returns_predictions_and_accuracy():
    features_train, target_train, features_test, target_test = _train_and_test_data()
    models = train_all_models(MODEL_CONFIG, features_train, target_train)

    result = evaluate(models["Random Forest"], features_test, target_test)

    assert "predictions" in result
    assert "accuracy" in result
    assert len(result["predictions"]) == len(target_test)
    assert 0 <= result["accuracy"] <= 1
    assert result["accuracy"] == accuracy_score(target_test, result["predictions"])


def test_evaluate_models_returns_best_model():
    features_train, target_train, features_test, target_test = _train_and_test_data()
    models = train_all_models(MODEL_CONFIG, features_train, target_train)

    best_model = evaluate_models(models, features_test, target_test)

    assert best_model["name"] in [name for name, _, _ in MODEL_CONFIG]
    assert 0 <= best_model["accuracy"] <= 1
    assert best_model["accuracy"] >= 0
