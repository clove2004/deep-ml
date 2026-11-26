# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(n*train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train = np.concatenate((np.ones((len(train_idx), 1)), X[train_idx]), axis=1)
    y_train = y[train_idx]
    weights, *_ = np.linalg.lstsq(X_train, y_train, rcond = None)
    
    X_test = np.concatenate((np.ones((len(test_idx), 1)), X[test_idx]), axis=1)
    test_predictions = (X_test @ weights).ravel()
    
    return test_predictions, train_idx, val_idx, test_idx