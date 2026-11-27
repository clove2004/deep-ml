import numpy as np
from typing import Tuple
def gini_impurity(y):
    n = len(y)
    dic = dict()
    for i in y:
        dic[i] = 0
    for i in y:
        dic[i] += 1
    vals = 1.0
    for key, val in dic.items():
        vals -= (val/n)**2
    return vals

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    n_samples, n_features = X.shape
    feature_index = -1
    threshold = -1
    G_cur = 1e10
    for j in range (n_features):
        thresholds = np.unique(X[:, j])
        for threshold_cur in thresholds:
            left, right = [], []
            for k in range (n_samples):
                if X[k][j] <= threshold_cur:
                    left.append(y[k])
                else:
                    right.append(y[k])
            G = (len(left)/n_samples)*gini_impurity(left) + (len(right)/n_samples)*gini_impurity(right)
            if G_cur > G:
                G_cur = G 
                threshold = threshold_cur
                feature_index = j
    return feature_index, threshold