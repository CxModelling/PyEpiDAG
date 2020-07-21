from epidag.fitting.alg.fitter import Fitter
import numpy as np

__author__ = 'TimeWz667'
__all__ = ['ABC']


class ABC(Fitter):
    DefaultParameters = dict(Fitter.DefaultParameters)
    DefaultParameters['n_test'] = 100
    DefaultParameters['p_test'] = 0.15
    DefaultParameters['pr_drop'] = 0.1
    DefaultParameters['n_update'] = 100

    def __init__(self, bm, **kwargs):
        Fitter.__init__(self, bm, **kwargs)
        self.Eps = None

    def fit(self, **kwargs):
        self.update_parameters(**kwargs)

        self.info('Testing threshold')

        self.initialise_prior(self['n_test'])

        tests = [p.LogLikelihood for p in self.Prior]
        self.Eps = np.percentile(tests, (1-self['p_test'])*100)

        self.Posterior.clear()

        self.info('Collecting posterior parameters')

        n = self['n_population']
        while len(self.Posterior) < n:
            self.add_a_posterior()
        self.info('Fitting completed')

    def update(self, **kwargs):
        self.update_parameters(**kwargs)
        self.info('Updating')
        n_target = len(self.Posterior) + self['n_update']

        while len(self.Posterior) < n_target:
            self.add_a_posterior()
        self.info('Update completed with {:d} posterior parameters'.format(len(self.Posterior)))

    def add_a_posterior(self):
        p = self.Model.sample_prior()
        li = self.Model.evaluate_likelihood(p)
        if li < self.Eps:
            return
        p.LogLikelihood = li
        self.Posterior.append(p)
