import numpy as np
def precision(y_true, y_pred):
    # Your code here
    TP, FP = 0, 0
    for i in range (len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
    precision = 0.0
    if (TP + FP) != 0:
        precision = TP/(TP + FP)
    return precision