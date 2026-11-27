import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n_samples = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    sigma = (1/n_samples)* X_centered.T @ X_centered
    eigval, eigvec = np.linalg.eigh(sigma)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]
    Vk = eigvec[:, :n_components]

    X_re = X_centered @ Vk @ Vk.T + np.mean(X, axis=0)
    d = len(eigval)

    mse = 0
    for i in range (n_samples):
        for j in range (d):
            mse += (X_re[i][j] - X[i][j])**2
    return mse/(n_samples*d)
