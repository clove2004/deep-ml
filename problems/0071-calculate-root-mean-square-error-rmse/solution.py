import numpy as np

def rmse(y_true, y_pred):
    # Write your code here
    if y_true.shape != y_pred.shape or len(y_true) == 0:
        raise ValueError()
    n = len(y_true)
    rmse_res = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return round(rmse_res,3)