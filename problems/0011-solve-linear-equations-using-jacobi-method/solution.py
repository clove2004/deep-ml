import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    lap = n 
    n = len(A)
    x = np.zeros(n, dtype=float)
    for _ in range (lap):
        x_new = np.zeros(n, dtype=float)
        for i in range (n):
            x_new[i] = b[i]
            for j in range (n):
                if i != j:
                    x_new[i] -= A[i][j]*x[j]
            x_new[i] /= A[i][i]
        x = x_new
    return x