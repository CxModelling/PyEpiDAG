from scipy.misc import logsumexp
from epidag import resample
from .fitter import Fitter

__author__ = 'TimeWz667'


class SIR(Fitter):
    def __init__(self, bm):
        Fitter.__init__(self, bm)
        self.Prior = list()

    def initialise(self):
        self.Posterior.clear()

    def fit(self, niter, **kwargs):
        self.Posterior.clear()
        self.Prior.clear()
        lis = list()
        for _ in range(niter):
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            p.LogLikelihood = li
            lis.append(li)
            self.Prior.append(p)

        self.Posterior, _ = resample(lis, self.Prior)
