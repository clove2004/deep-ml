def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.
    
    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """
    su = 0
    n = len(priors)
    for i in range (n):
        su += priors[i]*likelihoods[i]
    ans = []
    for i in range (n):
        ans.append(priors[i]*likelihoods[i]/su)
    return ans
    pass