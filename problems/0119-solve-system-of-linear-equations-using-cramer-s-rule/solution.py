import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.asarray(A)
    b = np.asarray(b)
    detA = np.linalg.det(A)
    n = len(A)
    if detA == 0:
        return -1
    X = np.zeros(n, dtype=float)
    for i in range (n):
        af = A.copy()
        for j in range (n):
            af[j][i] = b[j]
        X[i] = np.linalg.det(af)
    X /= detA 
    return X
    pass