from epidag import resample
from .fitter import BayesianFitter

__author__ = 'TimeWz667'
__all__ = ['SIR']


class SIR(BayesianFitter):
    def __init__(self, bm):
        BayesianFitter.__init__(self, bm)
        self.LogWts = None

    def initialise(self):
        self.Prior.clear()
        self.Posterior.clear()

    def fit(self, niter, **kwargs):
        self.info('Initialising')
        self.Posterior.clear()
        self.Prior.clear()
        lis = list()
        self.info('Sampling-Importance')
        for _ in range(niter):
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            p.LogLikelihood = li
            lis.append(li)
            self.Prior.append(p)

        self.LogWts = lis

        self.info('Resampling')
        self.Posterior, _ = resample(lis, self.Prior)

    def update(self, n_add, **kwargs):
        n = n_add + len(self.Posterior)
        self.Posterior, _ = resample(self.LogWts, self.Prior, new_size=n)
