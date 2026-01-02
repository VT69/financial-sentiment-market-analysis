import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_test, y_pred_log, asset="BTC", n=100):
    actual = np.exp(y_test.values[:n])
    predicted = np.exp(y_pred_log[:n])

    plt.figure(figsize=(12,4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted (RF)")
    plt.title(f"{asset} — Sentiment-Aware Volatility Forecast")
    plt.legend()
    plt.show()


def feature_importance(model, features):
    imp = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("\n📌 Feature Importance")
    print(imp)

    return imp
