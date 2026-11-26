import numpy as np

def rref(matrix):
    # Your code here
    matrix = np.asarray(matrix, dtype=float)
    m, n = matrix.shape
    rank = 0
    for col in range (n):
        if rank == m:
            break
        cur_row = rank + np.argmax(np.abs(matrix[rank:, col]))
        if np.abs(matrix[cur_row][col]) <= 1e-10:
            continue
        if cur_row != rank:
            a = matrix[rank].copy()
            matrix[rank] = matrix[cur_row]
            matrix[cur_row] = a 
        matrix[rank] /= matrix[rank][col]
        for row in range(m):
            if row != rank:
                fac = matrix[row][col]
                matrix[row, col:] -= fac * matrix[rank, col:]
        rank += 1
    return matrix
    pass
