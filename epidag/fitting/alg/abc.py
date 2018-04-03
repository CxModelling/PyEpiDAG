from epidag.fitting.alg.fitter import Fitter
import numpy as np
import logging

__author__ = 'TimeWz667'
__all__ = ['ABC']


logger = logging.getLogger(__name__)


class ABC(Fitter):
    DefaultParameters = {
        'test_n': 100,
        'test_p': 0.05
    }

    def __init__(self, bm):
        Fitter.__init__(self, bm)
        self.Prior = list()

    def initialise(self):
        self.Posterior.clear()

    def fit(self, niter, **kwargs):

        logger.info('Initialising')
        test_n = kwargs['test_n'] if 'test_n' in kwargs else ABC.DefaultParameters['test_n']
        test_p = kwargs['test_p'] if test_n in kwargs else ABC.DefaultParameters['test_p']
        test_p = max(test_p, 1/test_n)

        self.Posterior.clear()

        self.Prior = [self.Model.sample_prior() for _ in range(niter)]

        logger.info('Testing threshold')

        tests = list()
        for _ in range(test_n):
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            tests.append(li)
        eps = np.percentile(tests, (1-test_p)*100)

        logger.info('Fitting')
        while len(self.Posterior) < niter:
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            if li < eps:
                continue
            p.LogLikelihood = li
            self.Posterior.append(p)
        logger.info('Gathering posteriori')
