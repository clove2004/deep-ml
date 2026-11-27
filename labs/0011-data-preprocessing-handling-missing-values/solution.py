# --- cell 1 ---
import numpy as np
def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    # TODO: Fill in NaN values
    is_nan = np.isnan(X_clean)
    n_samples, n_features = X_clean.shape
    for j in range (n_features):
        c, sum = 0, 0
        for i in range (n_samples):
            if not is_nan[i][j]:
                c += 1
                sum += X_clean[i][j]
        mean = sum/c if c > 0 else 0.0
        for i in range (n_samples):
            if is_nan[i][j]:
                X_clean[i][j] = mean
    return X_clean