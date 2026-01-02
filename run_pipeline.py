import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.feature_engineering import build_features
from pipeline.train_rf_model import train_random_forest
from pipeline.evaluate import feature_importance, plot_predictions


def run(asset="BTC"):
    path = f"data/processed/{asset.lower()}_sentiment_aligned.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    X, y, features = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model, rmse, mae = train_random_forest(
        X_train, X_test, y_train, y_test, asset=asset
    )

    feature_importance(model, features)
    plot_predictions(y_test, model.predict(X_test), asset=asset, n = 100)


if __name__ == "__main__":
    run("BTC")
    # run("NIFTY")  # enable once ready-