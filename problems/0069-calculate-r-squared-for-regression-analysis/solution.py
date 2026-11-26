import numpy as np

def r_squared(y_true, y_pred):
    # Write your code here
    SS_res = np.sum((y_pred - y_true)**2)   
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    if SS_tot != 0:
        return 1.0 - SS_res/SS_tot
    else:
        if SS_res == 0:
            return 1.0
        else:
            return 0.0 
