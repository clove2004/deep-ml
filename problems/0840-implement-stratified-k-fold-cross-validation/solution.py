import numpy as np

def stratified_kfold_indices(y, n_splits):
    """
    Generate train/test indices for stratified K-fold cross-validation.

    Args:
        y: 1D array-like of integer class labels
        n_splits: number of folds

    Returns:
        A list of [train_indices, test_indices] pairs, one per fold.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    dic = dict()
    for c in classes:
        idc = np.where(y == c)[0].tolist()
        n = len(idc)
        sz = [(n//n_splits) for _ in range (n_splits)]
        r = n%n_splits
        for i in range (r):
            sz[i] += 1
        t = 0
        for i in range (n_splits):
            if i not in dic:
                dic[i] = []
            dic[i] += idc[t:t + sz[i]]
            t += sz[i]
    ans = []
    for i in range (n_splits):
        dic[i].sort()
        train_idc = [j for j in range (len(y)) if j not in dic[i]]
        ans.append([train_idc, dic[i]])
    return ans