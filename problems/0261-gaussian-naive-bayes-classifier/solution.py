import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)
    
    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    # Your code here
    eps = 1e-9
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    pos = []
    for c in classes:
        mean = np.mean(X_train[y_train == c], axis=0)
        var = np.mean((X_train[y_train == c] - mean)**2, axis=0) + eps
        py = X_train[y_train == c].shape[0]/n_samples
        P_test = (-1/2)*np.log(2*np.pi*var) - ((X_test - mean)**2)/(2*var)
        score = np.log(py) + np.sum(P_test, axis=1)
        pos.append(score)
    best_id = np.argmax(pos, axis=0)    
    return classes[best_id]
