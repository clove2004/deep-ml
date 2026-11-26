import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    acc = np.mean(y_true == y_pred)
    return acc