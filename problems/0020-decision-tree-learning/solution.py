import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    # Your code here
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count/n 
        entropy -= p*math.log2(p)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Your code here
    cur, D = {}, []
    for dic in examples:
        if dic[attr] in cur:
            cur[dic[attr]].append(dic[target_attr])
        else:
            cur[dic[attr]] = [dic[target_attr]]
        D.append(dic[target_attr])
    IG = calculate_entropy(D)
    n = len(D)
    for key, value in cur.items():
        n_0 = len(value)
        IG -= calculate_entropy(value)*(n_0/n)
    return IG

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    # Your code here
    if not examples:
        return ""
    d = []
    for dic in examples:
        d.append(dic[target_attr])
    counts = Counter(d)
    return min(counts.items(), key=lambda x: (-x[1], x[0]))[0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Your code here
    if not examples:
        return {}
    # end 1
    first_label = examples[0][target_attr]
    ok = True 
    for dic in examples:
        if dic[target_attr] != first_label:
            ok = False
            break
    if ok:
        return first_label
    # end 2
    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = max(
        attributes,
        key=lambda attr: calculate_information_gain(examples, attr, target_attr)
    )
    tree = {best_attr : {}}
    remain_attributes = [attr for attr in attributes if attr != best_attr]
    unique_val = sorted({dic[best_attr] for dic in examples})
    for val in unique_val:
        sub_example = [dic for dic in examples if dic[best_attr] == val]
        if not sub_example:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            tree[best_attr][val] = learn_decision_tree(sub_example, remain_attributes, target_attr)
    return tree