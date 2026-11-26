import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    for i in range (len(std)):
        if std[i] == 0.0:
            std[i] = 1.0
    X_test = (X_test - mean)/std
    return X_test
    pass
