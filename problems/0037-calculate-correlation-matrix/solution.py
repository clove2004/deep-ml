import numpy as np
def calculate_correlation_matrix(X, Y=None):
    # Your code here
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    rowx = np.mean(X, axis = 0)
    rowy = np.mean(Y, axis = 0)
    stdx = np.std(X, axis=0, ddof=1)
    stdy = np.std(Y, axis=0, ddof=1)
    n_sample, d1 = X.shape
    _, d2 = Y.shape
    ans = np.zeros((d1, d2), dtype = float)
    for i in range (d1):
        for j in range (d2):
            if stdx[i] == 0 or stdy[j] == 0:
                ans[i, j] = 0.0
                continue
            for k in range (n_sample): 
                ans[i][j] += (X[k][i] - rowx[i])*(Y[k][j] - rowy[j])
            ans[i][j] /= (n_sample - 1)*stdx[i]*stdy[j]
    return ans.tolist()
    pass