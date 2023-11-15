"""
File for keeping relevent information
"""
import numpy as np

distributions = {
    'normal' : {
        'MLE' : {
            'mu' : np.mean,
            'sigma' : lambda x: np.std(x, ddof=0)
        }
    },
    'exponential': {
      'MLE' : {
        'lambda' : lambda x: 1/np.mean(x)
      }
    },
    'weibull' : {},
    'gamma' : {},
    'beta' : {},
    'lognormal' : {},
    'cauchy' : {}
}