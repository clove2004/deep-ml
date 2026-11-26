from typing import Callable
import numpy as np
def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    # Your code here
    n = len(point)
    point = np.asarray(point)
    E = np.eye(n, dtype=float)
    e = []
    for i in range (n):
        e.append(E[i, :])
    Hes = np.zeros((n, n), dtype=float)
    for i in range (n):
        for j in range (n):
            if i == j:
                tuso = (f(point + h*e[i]) - 2*f(point) + f(point - h*e[i]))
                Hes[i][j] = tuso/(h**2)
            else:
                tuso = (f(point + h*e[i] + h*e[j]) + f(point - h*e[i] - h*e[j]) - f(point + h*e[i] - h*e[j]) - f(point - h*e[i] + h*e[j]))
                Hes[i][j] = tuso/(4*(h**2))
    return Hes.tolist()
    pass