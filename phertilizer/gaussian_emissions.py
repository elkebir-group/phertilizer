import numpy as np
from scipy.stats import multivariate_normal


class RDRGaussianMixture():
    """
    A class to model the emission distributions of CNA genotypes as Gaussians

    ...

    Attributes
    ----------
    neutral_mean : float
        the parameter specifying the approximate location of the neutral state 
        used to infer the emission distribution from the rdr data
    neutral_eps : float
        the parmaeter specifying the approximate standard deviation of the neutral state
        used to the emission distribution from the rdr data
    gaussian_mixture : dict
        a dictionary with states as keys containing a Gaussian emission distribution

    Methods
    -------

    fit(X)
        fits three Gaussian distributions (one for each state) from the given input data X

    compute_log_pdf(state, point)
        computes the log probability of the given points under a specified state

    """

    def __init__(self, neutral_mean=1.0, eps=0.15):
        self.gaussian_mixture = {}

        self.neutral_mean = float(neutral_mean)
        self.eps = float(eps)

    def fit(self, X):
        '''fits three Gaussian distributions (one for each state) from the given input data X

        Parameters
        ----------
        X : np.array
            an np.array of RDR values for which the emission distributions should be fit

        '''

        X = X.reshape(-1, 1)

        cut1 = self.neutral_mean - 2*self.eps
        cut2 = self.neutral_mean + 2*self.eps
        loss_bins = X[X < cut1]
        norm_bins = X[np.logical_and(X >= cut1, X <= cut2)]
        gain_bins = X[X > cut2]

        self.default_var = 1

        if len(loss_bins) > 1:
            self.gaussian_mixture['loss'] = multivariate_normal(
                mean=np.mean(loss_bins), cov=np.var(loss_bins))
        else:
            self.gaussian_mixture['loss'] = multivariate_normal(
                mean=cut1-2*self.eps, cov=self.default_var)

        if len(gain_bins) > 1:
            self.gaussian_mixture['gain'] = multivariate_normal(
                mean=np.mean(gain_bins), cov=np.var(gain_bins))
        else:
            self.gaussian_mixture['gain'] = multivariate_normal(
                mean=cut2+2*self.eps, cov=self.default_var)

        if len(norm_bins) > 1:
            self.gaussian_mixture['neutral'] = multivariate_normal(
                mean=np.mean(norm_bins), cov=np.var(norm_bins))
        else:
            self.gaussian_mixture['neutral'] = multivariate_normal(
                mean=self.neutral_mean, cov=self.default_var)

    def compute_log_pdf(self, state, point):
        '''computes the log probability of the given points under a specified state

        Parameters
        ----------
        state : str
           the state for which the log emmission probability should be computed
        point : np.array or float
           the observed data for which the log emmission probability should be computed 

        '''
        return self.gaussian_mixture[state].logpdf(point)
