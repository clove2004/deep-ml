import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    X_cur = (X - mean)
    sigma = (1/(n_samples - 1)) * X_cur.T @ X_cur
    eigval, eigvec = np.linalg.eigh(sigma)
    eigval = eigval[::-1]
    sum_val = np.sum(eigval)
    return eigval/sum_val