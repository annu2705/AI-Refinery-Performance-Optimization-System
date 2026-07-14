"""
train.py
Model training entry point. Each module (Boiler, Throughput, Energy, Yield)
can call these functions with its own target column and data path.
"""

import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

try:
    from src.preprocessing import load_data, clean_data, engineer_features
except ImportError:
    from preprocessing import load_data, clean_data, engineer_features


def train_model(data_path: str, target_col: str, model_out_path: str):
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(verbose=False, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_out_path)
    print(f"Model saved to {model_out_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    # Example usage — update per module
    train_model(
        data_path="data/raw_dataset.csv",
        target_col="Boiler_Eff_pct",
        model_out_path="models/boiler_model.pkl",
    )
