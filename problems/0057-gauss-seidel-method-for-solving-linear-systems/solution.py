import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    lap = n 
    n = len(A)
    if x_ini is None:
        x_ini = np.zeros(n, dtype=float)
    for _ in range(lap):
        for i in range (n):
            x_ini[i] = b[i]
            for j in range (i):
                x_ini[i] -= x_ini[j]*A[i][j]
            for j in range (i + 1, n):
                x_ini[i] -= x_ini[j]*A[i][j]
            x_ini[i] /= A[i][i]
    return x_ini
