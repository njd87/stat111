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



    'weibull' : {

        'MLE' : {
            'lambda' : lambda data, gamma: 1/(np.mean(data ** gamma))
        }

    },



    'gamma' : {

        'MLE' : {
            'beta' : lambda data, alpha: np.mean(data)/alpha
        }

    },



    'beta' : {

        'MLE' : {}

    },



    'lognormal' : {

        'MLE' : {
            'mu' : lambda x: np.mean(np.log(x)),
            'sigma' : lambda x: np.std(np.log(x), ddof=0)
        }

    },
}