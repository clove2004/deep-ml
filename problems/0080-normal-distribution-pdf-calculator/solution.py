import math
import numpy as np
def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    # Your code here
    var = std_dev**2
    val = ((2*math.pi)**(-0.5))*(var**(-0.5))*np.exp(-0.5*(x-mean)*(var**(-1))*(x-mean))
    return round(val,5)