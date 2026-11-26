import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    ans = np.zeros((n, n), dtype=float)
    for i in range (n):
        for j in range (n):
            if np.linalg.norm(X[i, :]) != 0 and np.linalg.norm(X[j, :]) != 0:
                ans[i][j] = (X[i, :]@X[j, :])/(np.linalg.norm(X[i, :])*np.linalg.norm(X[j, :]))
    return ans.tolist()
    pass