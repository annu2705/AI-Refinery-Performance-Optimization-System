"""
predict.py
Loads a saved model and runs predictions on new data.
Used by the Streamlit dashboard for live inference.
"""

import joblib
import pandas as pd


def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model_path: str, input_df: pd.DataFrame) -> pd.Series:
    model = load_model(model_path)
    return model.predict(input_df)


if __name__ == "__main__":
    # Example usage
    sample = pd.DataFrame(
        {
            "Furnace_Temp_C": [350],
            "Reactor_Temp_C": [420],
            "Reflux_Ratio": [2.1],
            "Pressure_Drop_bar": [1.5],
        }
    )
    result = predict("models/boiler_model.pkl", sample)
    print(result)
