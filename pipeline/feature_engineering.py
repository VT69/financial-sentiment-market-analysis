import numpy as np

def build_features(df, target_horizon=5):
    """
    df must contain:
    - date
    - return
    - finbert_score
    - vader_score
    """

    df = df.sort_values("date").reset_index(drop=True)

    # Volatility memory
    df["vol_5"] = df["return"].rolling(5).std()
    df["vol_22"] = df["return"].rolling(22).std()
    df["vol_60"] = df["return"].rolling(60).std()

    # Target: next-day volatility
    df["vol_target"] = df["return"].rolling(5).std().shift(-target_horizon)
    df["log_vol_target"] = np.log(df["vol_target"] + 1e-6)

    # Price dynamics
    df["abs_return"] = np.abs(df["return"])
    df["sq_return"] = df["return"] ** 2

    # Sentiment surprise
    df["finbert_surprise"] = (
        df["finbert_score"] - df["finbert_score"].rolling(5).mean()
    )
    df["vader_surprise"] = (
        df["vader_score"] - df["vader_score"].rolling(5).mean()
    )

    # Sentiment interactions
    df["sent_x_vol5"] = df["finbert_score"] * df["vol_5"]
    df["sent_x_vol22"] = df["finbert_score"] * df["vol_22"]
    df["sent_x_absret"] = df["finbert_score"] * df["abs_return"]

    df = df.dropna().reset_index(drop=True)

    FEATURES = [
        "finbert_score", "vader_score",
        "finbert_surprise", "vader_surprise",
        "abs_return", "sq_return",
        #"vol_5", "vol_22", "vol_60",
        "sent_x_vol5", "sent_x_vol22", "sent_x_absret"
    ]

    X = df[FEATURES]
    y = df["log_vol_target"]

    print("Target variance:", y.var())
    print(y.describe())

    return X, y, FEATURES
