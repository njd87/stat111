import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from constants import distributions


def likelihood_iid(data: any, likelihood) -> float:
    """
    Calculate the likelihood of a given array of iid data
    
    Parameters:
    -----------
    data : any
        The data to calculate the likelihood for. Should be an interable object (list, np.array, etc.)
        ***Try to keep data under around 200 observations for general purpose simulation.***
    likelihood: function
        The likelihood function should have the following structure:

        def l(y, theta):
            return ...
        
        With "y" the parameter of the observed value and "theta" the paramter of the estimand
        
    Returns:
    --------
    float
        The likelihood of the specified parameter
    """

    # create a new function to maximize over theta

    def likelihood_max(theta):
        s = 1

        for observation in data:
            s *= likelihood(observation, theta)
        
        return s
    
    # use scipy.optimize to find the maximum of likelihood_max
    # return the arg min (that is, return the "theta" that generates the maximum value)
    theta = minimize_scalar(lambda theta: -likelihood_max(theta))
    return theta.x
    

def likelihood_common(data: any, distribution: str, parameters: list):
    """
    Calculate the likelihood from common distributions. Distributions and parameters include:

    normal -- mu, sigma
    exponential -- lambda
    weibull -- lambda, gamma
    gamma -- alpha, lambda
    beta -- alpha, beta
    lognormal -- mu, sigma
    cauchy -- loc, gamma

    Parameters:
    ---------

    data: any
        The data to calculate the likelihood for. Should be an interable object (list, np.array, etc.)

    distribution: name of common distribution

    parameters: list
        List of parameters from the distribution you wish to estimate
    """

    # check to make sure distribution is valid
    if distribution not in distributions:
        raise ValueError('Distribution not included')
    
    # create result dictionary
    res = {}

    # iterate through parameters
    for parameter in parameters:
        # check if parameter is valid
        if parameter not in distributions[distribution]['MLE']:
            raise ValueError(f'{parameter} not found for distribution {distribution}')
        
        res[parameter] = distributions[distribution]['MLE'][parameter](data)

    return res