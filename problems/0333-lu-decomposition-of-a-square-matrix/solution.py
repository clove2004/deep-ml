import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    
    Args:
        A: Square matrix as a list of lists
    
    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    # Your code here
    A = np.asarray(A)
    n = len(A)
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range (n):
        for j in range (n):
            if j >= i:
                U[i][j] = A[i][j]
                for k in range (i):
                    U[i][j] -= L[i][k]*U[k][j]
            if j > i:
                L[j][i] = A[j][i]
                for k in range (i):
                    L[j][i] -= L[j][k]*U[k][i]
                if U[i][i] == 0:
                    L[j][i] = 0
                else:
                    L[j][i] /= U[i][i]
    return (L, U)
    pass