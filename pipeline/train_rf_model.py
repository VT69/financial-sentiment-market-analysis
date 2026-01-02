import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_random_forest(X_train, X_test, y_train, y_test, asset="BTC"):
    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    pred_vol = np.exp(preds)
    actual_vol = np.exp(y_test)

    rmse = np.sqrt(mean_squared_error(actual_vol, pred_vol))
    mae = mean_absolute_error(actual_vol, pred_vol)

    print(f"\n📈 {asset} Random Forest Results")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model, model_dir / f"rf_{asset.lower()}.pkl")

    return model, rmse, mae
