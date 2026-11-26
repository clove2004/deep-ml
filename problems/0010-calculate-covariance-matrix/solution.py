import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    # Your code here
    vec = np.asarray(vectors)
    row = np.mean(vec, axis = 1)
    n, m = vec.shape
    if n == 1:
        return [[1.0]]
    ans = np.zeros((n, n), dtype = float)
    for i in range (n):
        for j in range (n):
            for k in range (n):
                ans[i][j] += (vec[i][k] - row[i])*(vec[j][k] - row[j])
            ans[i][j] /= (n - 1)
    return ans.tolist()