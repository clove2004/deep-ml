import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    theta = np.linalg.inv((np.transpose(X)@X)) @ np.transpose(X)
    theta = theta @ y
    return theta