"""
evaluate.py
Model evaluation utilities — used for QA/testing (Annu) to verify each
module's model meets baseline performance before dashboard integration.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_model(model, X_test, y_test) -> dict:
    """Return standard regression metrics for a trained model."""
    preds = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    print("Run this after training a model — import and call evaluate_model().")
