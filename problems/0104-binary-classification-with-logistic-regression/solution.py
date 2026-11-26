import numpy as np
def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Your code here
    y_pred = X @ weights + bias
    y_sig = 1 / (1 + np.exp(-y_pred))
    ans = []
    for i in y_sig:
        if i >= 0.5:
            ans.append(1)
        else:
            ans.append(0)
    return ans