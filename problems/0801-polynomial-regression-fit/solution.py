import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=float)
    n = len(x)
    vander_matrix = np.ones((n, degree+1), dtype=float)
    for i in range (n):
        for j in range (1, degree + 1):
            vander_matrix[i][j] = vander_matrix[i][j - 1]*x[i]
    theta, *_ = np.linalg.lstsq(vander_matrix, y)
    return theta

