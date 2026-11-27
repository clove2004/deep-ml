# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    num_interations = 1500
    lr = 0.01
    n = X.shape[0]
    for _ in range (num_interations):
        er = X @ W + b - y
        grad_w = (2/n) * (X.T @ er)
        grad_b = (2/n) * np.sum(er)
        W = W - lr * grad_w
        b = b - lr * grad_b

    return (W, b)
    pass