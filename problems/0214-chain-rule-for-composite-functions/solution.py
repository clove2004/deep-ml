import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # Your code here
    opts = {
        'sin': (np.sin, np.cos),
        'exp': (np.exp, np.exp),
        'log': (np.log, lambda u: 1.0/u),
        'square': (lambda u: u**2, lambda u: 2*u)
    }
    cur_val = x
    ans = 1.0
    for name in reversed(functions):
        f, df = opts[name]
        ans *= df(cur_val)
        cur_val = f(cur_val)
    return ans
    pass