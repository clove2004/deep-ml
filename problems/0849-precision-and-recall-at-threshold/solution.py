import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    TP, FN, FP = 0, 0, 0
    for i in range (len(y_true)):
        y_pred = 0
        if y_scores[i] >= threshold:
            y_pred = 1
        if y_pred == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred == 0 and y_true[i] == 1:
            FN += 1
        elif y_pred == 1 and y_true[i] == 0:
            FP += 1
    recall, precision = 0.0, 0.0
    if (TP + FN) != 0:
        recall = TP/(TP + FN)
    if (TP + FP) != 0:
        precision = TP/(TP + FP)
    recall = round(recall, 4)
    precision = round(precision, 4)
    return [precision, recall]