import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    counts = Counter(y_train)
    if strategy == 'most_frequent':
        ans, mx = counts.most_common(1)[0]
        for key, val in counts.items():
            if val == mx and ans > key:
                ans = key
        return [ans for _ in range (n_test)]
    elif strategy == 'constant':
        return [constant for _ in range (n_test)]
    elif strategy == 'uniform':
        lst = []
        for key, value in counts.items():
            lst.append(key)
        lst.sort()
        k = len(lst)
        return [lst[i%k] for i in range (n_test)]
    else:
        summ, tol = 0, 0
        dic = dict()
        for key, val in counts.items():
            summ += val 
        for key, val in counts.items():
            nu = n_test*(val/summ)
            dic[key] = (int(nu), nu - int(nu))
            tol += int(nu)
        rem = n_test - tol
        sorted_d = dict(sorted(dic.items(), key=lambda x: (-x[1][1], x[0])))
        ans = []
        for key, val in sorted_d.items():
            if rem > 0:
                rem -= 1
                ans.extend([key] * (val[0] + 1))
            else:
                ans.extend([key] * (val[0]))
        ans.sort()
        return ans