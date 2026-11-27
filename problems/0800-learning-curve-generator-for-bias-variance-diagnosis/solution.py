import numpy as np

def build_features(x, degree):
    x = np.asarray(x,dtype=float).reshape(-1)
    n = len(x)
    vander_matrix = np.ones((n, degree+1), dtype=float)
    for i in range (n):
        for j in range (1, degree + 1):
            vander_matrix[i][j] = vander_matrix[i][j - 1]*x[i]
    return vander_matrix

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    train_errors, val_errors = [], []
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    Phi_train = build_features(X_train, degree)
    Phi_val = build_features(X_val, degree)

    for n in train_sizes:
        Phi_train_n = Phi_train[:n]
        y_train_n = y_train[:n]

        w = np.linalg.pinv(Phi_train_n) @ y_train_n
        y_train_pred = Phi_train_n @ w 
        y_val_pred = Phi_val @ w 

        train_e = np.mean((y_train_pred - y_train_n)**2)
        val_e = np.mean((y_val_pred - y_val)**2)
        train_errors.append(train_e)
        val_errors.append(val_e)
    train_end = train_errors[len(train_errors) - 1]
    val_end = val_errors[len(val_errors) - 1]
    s = ""
    if train_end > bias_threshold:
        s = 'high_bias'
    elif val_end - train_end > variance_threshold:
        s = 'high_variance'
    else:
        s = 'good_fit'
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': s
    }