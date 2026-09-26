from configs.config import MODEL_CONFIG, TARGET_COL
from src.data import clean_data, load_data
from src.features import build_features
from src.model import train_all_models
from src.predict import predict_species


def _prepare_prediction_inputs():
    df = clean_data(load_data("data/penguins.csv"))
    features, encoders = build_features(df)
    target = df[TARGET_COL]

    train_size = len(df) - 10
    train_features = features.iloc[:train_size]
    train_target = target.iloc[:train_size]

    model = train_all_models(MODEL_CONFIG, train_features, train_target)[
        "Random Forest"
    ]
    _, _, scaler = __import__(
        "src.features", fromlist=["scale_features"]
    ).scale_features(train_features, train_features)

    row = df.iloc[0].to_dict()
    return row, model, scaler, encoders, list(train_features.columns)


def test_predict_species_returns_species_name():
    sample, model, scaler, encoders, feature_cols = _prepare_prediction_inputs()

    prediction = predict_species(sample, model, scaler, encoders, feature_cols)

    assert prediction in set(clean_data(load_data("data/penguins.csv"))[TARGET_COL])
