from configs.config import MODEL_CONFIG, TARGET_COL
from src.data import clean_data, load_data
from src.features import build_features
from src.model import train_all_models, train_model


def _train_data():
    df = clean_data(load_data("data/penguins.csv"))
    features, _ = build_features(df)
    target = df[TARGET_COL]
    train_size = len(df) - 10
    features_train = features.iloc[:train_size]
    target_train = target.iloc[:train_size]

    return features_train, target_train


def test_train_model_returns_fitted_estimator():
    features_train, target_train = _train_data()

    model_name, model_class, model_params = MODEL_CONFIG[0]
    model = model_class(**model_params)

    fitted_model = train_model(model, features_train, target_train)

    assert fitted_model is not None
    assert hasattr(fitted_model, "predict")


def test_train_all_models_returns_one_model_per_config_entry():
    features_train, target_train = _train_data()

    models = train_all_models(MODEL_CONFIG, features_train, target_train)

    assert list(models.keys()) == [name for name, _, _ in MODEL_CONFIG]
    assert len(models) == len(MODEL_CONFIG)
