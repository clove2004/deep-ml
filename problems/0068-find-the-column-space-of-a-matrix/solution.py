
import numpy as np

def matrix_image(A):
    # Write your code here
    A = np.asarray(A, dtype=float)
    M = A.copy()
    m, n = M.shape
    rank = 0
    ansid = []
    for col in range (n):
        if rank == m:
            break
        cur_row = rank + np.argmax(np.abs(M[rank:, col]))
        if np.abs(M[cur_row][col]) <= 1e-10:
            continue
        if cur_row != rank:
            a = M[rank].copy()
            M[rank] = M[cur_row]
            M[cur_row] = a 
        for row in range (rank + 1, m):
            fac = M[row][col]/M[rank][col]
            M[row, col:] -= fac*M[rank, col:]
        rank += 1
        ansid.append(col)
    ans = []
    for col in ansid:
        ans.append(A[:, col])
    ans = np.asarray(ans)
    ans = np.transpose(ans)
    return ans
    pass
