
from collections import Counter

def confusion_matrix(data):
    # Implement the function here
    TP = FN = FP = TN = 0
    for y_true, y_pred in data:
        if y_pred == 1 and y_true == 1:
            TP += 1
        elif y_pred == 1 and y_true == 0:
            FP += 1
        elif y_pred == 0 and y_true == 0:
            TN += 1
        elif y_pred == 0 and y_true == 1:
            FN += 1
    return [[TP, FN], [FP, TN]]