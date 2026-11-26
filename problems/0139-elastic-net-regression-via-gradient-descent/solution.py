import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0
    for _ in range(max_iter):
        res = X @ w + b - y 
        delta_w = (1/n_samples)*X.T @ res + alpha1 * np.sign(w) + 2 * alpha2 * w 
        delta_b = (1/n_samples)*np.sum(res)
        w = w - learning_rate*delta_w
        b = b - learning_rate*delta_b
        if np.linalg.norm(delta_w, ord=1) < tol:
            break
    return w, b