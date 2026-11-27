import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mean = np.mean(predictions, axis=0)
    bias_squared = (mean - y_true)**2
    var = np.mean((predictions - mean)**2, axis=0)

    b = np.mean(bias_squared)
    vr = np.mean(var)
    mse = b + vr 
    return {
        'bias_squared': b,
        'variance': vr,
        'mse': mse
    }