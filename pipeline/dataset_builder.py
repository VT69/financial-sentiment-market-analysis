# pipeline/dataset_builder.py

from sklearn.model_selection import train_test_split

FEATURES = [
    "finbert_score",
    "vader_score",
    "abs_return",
    "sq_return",
    "vol_5",
    "vol_22",
    "vol_60",
    "sent_vol"
]

TARGET = "log_vol"

def build_dataset(df, test_size=0.2):
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
