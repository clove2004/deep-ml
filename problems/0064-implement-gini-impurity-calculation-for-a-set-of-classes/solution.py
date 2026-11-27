
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    n = len(y)
    dic = dict()
    for i in y:
        dic[i] = 0
    for i in y:
        dic[i] += 1
    vals = 1.0
    for key, val in dic.items():
        vals -= (val/n)**2
    return round(vals,3)