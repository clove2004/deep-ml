import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    # Your code here
    n, d = X.shape
    bias = np.ones((n, 1), dtype=float)
    X = np.concatenate((bias, X), axis=1)
    weights = np.zeros(d + 1, dtype=float)
    lossed = []
    for _ in range (iterations):
        y_pred = 1 / (1 + np.exp(-(X @ weights)))
        L = 0
        for i in range (n):
            L -= y[i]*np.log(y_pred[i]) + (1 - y[i])*np.log(1 - y_pred[i])
        lossed.append(round(float(L), 4))
        weights = weights - learning_rate*(X.T @ (y_pred - y))
    weights = [round(float(c), 4) for c in weights]
    return weights, lossed