import numpy as np
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.
    
    Args:
        parameters: List of parameter arrays
        gradients: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    
    Returns:
        Updated parameters
    """
    # Your code here
    parameters = np.asarray(parameters, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    for i in range (len(apply_to_all)):
        if apply_to_all[i] == True:
            parameters[i] = parameters[i] - lr*gradients[i] - lr*weight_decay*parameters[i]
        else:
            parameters[i] = parameters[i] - lr*gradients[i]
    return parameters.tolist()