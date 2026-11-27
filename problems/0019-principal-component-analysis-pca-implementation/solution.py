import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    n_samples, n_features = data.shape
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.where(std == 0, 1.0, std)
    X = (data - mean)/std
    sigma = (1/(n_samples - 1))* X.T @ X
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    for j in range (k):
        for i in range (n_features):
            if np.abs(eigenvectors[i][j]) > 1e-10:
                if eigenvectors[i][j] < 0:
                    eigenvectors[:, j] *= -1
                break
    return np.round(eigenvectors[:, :k], 4)
