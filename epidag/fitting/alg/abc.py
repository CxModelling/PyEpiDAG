from .fitter import BayesianFitter
import numpy as np

__author__ = 'TimeWz667'
__all__ = ['ABC']


class ABC(BayesianFitter):
    DefaultParameters = {
        'test_n': 100,
        'test_p': 0.15,
        'max_drop': 1000
    }

    def __init__(self, bm):
        BayesianFitter.__init__(self, bm)
        self.TestN = ABC.DefaultParameters['test_n']
        self.TestP = ABC.DefaultParameters['test_p']
        self.MaxDrop = ABC.DefaultParameters['max_drop']
        self.Eps = None

    def initialise(self):
        self.Posterior.clear()

    def fit(self, niter, log=True, **kwargs):
        self.info('Initialising')
        if 'test_n' in kwargs:
            self.TestN = kwargs['test_n']
        if 'test_p' in kwargs:
            self.TestP = min(max(kwargs['test_p'], 1/self.TestN), 1)

        self.Posterior.clear()
        self.Prior = [self.Model.sample_prior() for _ in range(niter)]

        self.info('Testing threshold')

        if 'eps' in kwargs:
            self.Eps = kwargs['eps']
        else:
            drop = 0
            tests = list()
            for _ in range(self.TestN):
                p = self.Model.sample_prior()

                li = self.Model.evaluate_likelihood(p)
                if np.isfinite(li):
                    tests.append(li)
                else:
                    drop += 1
                    if drop > self.MaxDrop:
                        self.error('Too many infinite likelihood')
                        return

            self.Eps = np.percentile(tests, (1-self.TestP)*100)

        if not np.isfinite(self.Eps):
            self.error('Non-reasonable eps')
            return

        self.info('Fitting')
        while len(self.Posterior) < niter:
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            if li < self.Eps:
                continue

            p.LogLikelihood = li
            self.Posterior.append(p)
        self.info('Gathering posteriori')
        self.info('Finished')

    def update(self, n, **kwargs):
        self.info('Updating')
        n_target = len(self.Posterior) + n

        while len(self.Posterior) < n_target:
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            if li < self.Eps:
                continue
            p.LogLikelihood = li
            self.Posterior.append(p)
