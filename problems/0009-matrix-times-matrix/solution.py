import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    n, m, k = len(a), len(a[0]), len(b[0])
    if m != len(b):
        return -1
    c = np.zeros((n, k), dtype = int)
    for i in range (n):
        for j in range (k):
            for l in range (m):
                c[i][j] += a[i][l]*b[l][j]
    return c.tolist()