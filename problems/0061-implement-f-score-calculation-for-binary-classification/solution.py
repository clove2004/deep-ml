import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: Weight of recall relative to precision (beta > 1 favours recall)
    :return: F-Score rounded to three decimal places
    """
    TP, FN, FP = 0, 0, 0
    for i in range (len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        
    recall, precision = 0.0, 0.0
    if (TP + FN) != 0:
        recall = TP/(TP + FN)
    if (TP + FP) != 0:
        precision = TP/(TP + FP)
    denom = (beta ** 2) * precision + recall
    if denom == 0:
        return 0.0
    score = (1 + beta ** 2) * (precision * recall) / denom
    return round(score, 3)