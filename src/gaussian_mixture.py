import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


class RDRGaussianMixture():
    def __init__(self, neutral_mean=1.0, eps=0.15):
        self.gaussian_mixture = {}
        self.weights = {}
        self.neutral_mean = float(neutral_mean)
     

        self.eps = float(eps)

    
    def fit(self, X):
        X = X.reshape(-1,1)
       

        cut1 = self.neutral_mean - 2*self.eps
        cut2= self.neutral_mean +2*self.eps
        loss_bins = X[X < cut1]
        norm_bins = X[np.logical_and(X >= cut1, X <= cut2)]
        gain_bins = X[X > cut2]

        self.default_var=1




        
        if len(loss_bins) > 1:
            self.gaussian_mixture['loss'] = multivariate_normal(mean=np.mean(loss_bins), cov=np.var(loss_bins))
        else:
            self.gaussian_mixture['loss'] = multivariate_normal(mean=cut1-2*self.eps,cov = self.default_var)
        
        if len(gain_bins) > 1:
            self.gaussian_mixture['gain'] = multivariate_normal(mean=np.mean(gain_bins), cov=np.var(gain_bins))
        else:
            self.gaussian_mixture['gain'] = multivariate_normal(mean=cut2+2*self.eps,cov = self.default_var)
        
        if len(norm_bins) > 1:
            self.gaussian_mixture['neutral'] = multivariate_normal(mean=np.mean(norm_bins), cov=np.var(norm_bins))
        else:
            self.gaussian_mixture['neutral'] = multivariate_normal(mean=self.neutral_mean, cov = self.default_var)

       
 
    def pdf(self, point):
        return {key: self.gaussian_mixture[key].logpdf(point) for key in self.gaussian_mixture}
    

    def compute_log_pdf(self, state, point):
        return self.gaussian_mixture[state].logpdf(point)
    

    def loglikelihood(self, X, events):
        log_vals = []
        for key in events:
            dat = X[:,events[key]].reshape(-1,1)
            loglike_event = np.log(self.weights[key]) + self.gaussian_mixture[key].logpdf(dat)
            log_vals.append(loglike_event.astype(float))
        
        loglike = logsumexp(np.concatenate(log_vals))
    
        return loglike

      






    




