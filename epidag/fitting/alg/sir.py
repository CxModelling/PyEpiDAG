from epidag import resample
from epidag.fitting.alg.fitter import Fitter
import logging

__author__ = 'TimeWz667'
__all__ = ['SIR']


logger = logging.getLogger(__name__)


class SIR(Fitter):
    def __init__(self, bm):
        Fitter.__init__(self, bm)
        self.Prior = list()

    def initialise(self):
        self.Posterior.clear()

    def fit(self, niter, **kwargs):
        logger.info('Initialising')
        self.Posterior.clear()
        self.Prior.clear()
        lis = list()
        logger.info('Sampling-Importance')
        for _ in range(niter):
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            p.LogLikelihood = li
            lis.append(li)
            self.Prior.append(p)

        logger.info('Resampling')
        self.Posterior, _ = resample(lis, self.Prior)
