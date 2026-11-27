import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    ans = []
    n_samples = X.shape[0]
    for i in range (n_samples):
        anscur = [1]
        for deg in range (1, degree + 1):
            res = combinations_with_replacement(X[i], deg)
            for tup in res:
                c = 1.0
                for v in tup:
                    c *= v
                anscur.append(c)
        anscur.sort()
        ans.append(anscur)
    return np.array(ans)