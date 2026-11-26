import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """ 
    n_samples = X.shape[0]
    for i in range (n_iter):
        x_cur = X[i%n_samples]
        y_cur = y[i%n_samples]
        y_pred = x_cur @ weights
        weights = weights - 2*learning_rate*(y_pred - y_cur)*x_cur.T
    return weights.tolist()
    pass
