import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g_coeffs = np.asarray(g_coeffs)
    h_coeffs = np.asarray(h_coeffs)
    g_der = np.polyder(g_coeffs, 1)
    h_der = np.polyder(h_coeffs, 1)
    return (np.polyval(g_der, x)*np.polyval(h_coeffs, x) - np.polyval(h_der, x)*np.polyval(g_coeffs, x))/(np.polyval(h_coeffs, x)**2)

    # Your code here
    pass