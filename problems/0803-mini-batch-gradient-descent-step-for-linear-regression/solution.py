import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_new = X[batch_indices]
    y_new = y[batch_indices]
    m = len(batch_indices)
    y_pred = X_new @  weights + bias
    weights = weights - lr*(2/m)* (X_new.T @ (y_pred - y_new))
    bias = bias - lr*(2/m)*np.sum(y_pred - y_new)
    return np.concatenate([weights, [bias]])
