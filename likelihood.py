import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from constants import distributions


"""
TODO: Add docstring
"""
class MLE():
    
    def __init__(self, data: any):
        self.data = data
        self.estimate = None

        # TODO: scale data to allow for better predictions

    def iid(self, likelihood) -> float:
        """
        Calculate the MLE of a given array of iid data
        ***Try to keep data under around 100 observations for general purpose simulation.***
        
        Parameters:
        -----------
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

            for observation in self.data:
                s *= likelihood(observation, theta)
            
            return s
        
        # use scipy.optimize to find the maximum of likelihood_max
        # return the arg min (that is, return the "theta" that generates the maximum value)
        theta = minimize_scalar(lambda theta: -likelihood_max(theta))
        self.estimate = theta.x
        return self.estimate
    
    def iid_common(self, distribution: str, parameter: str, given: float = None) -> float:
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
        distribution: name of common distribution

        parameters: list
            parameter you wish to estimate

        given: float, default None
            For the gamma and weibull distributions, there is only closed form solutions for beta and lambda respectively.
            So, provide:

            alpha -> Gamma
            gamma -> Weibull
        """
        # check to make sure distribution is valid
        if distribution not in distributions:
            raise ValueError('Distribution not included')
    

        # check if parameter is valid
        if parameter not in distributions[distribution]['MLE']:
            raise ValueError(f'{parameter} not found for MLE estimation in distribution {distribution}')
        
        if distribution in {'gamma', 'weibull'}:
            if given is None: raise ValueError(f'{parameter} estimation for {distribution} requires given (see documentation)')
            return distributions[distributions]['MLE'][parameter](self.data, given)

        return distributions[distribution]['MLE'][parameter](self.data)