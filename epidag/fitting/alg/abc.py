from epidag.fitting import BayesResult
from epidag.fitting.alg.fitter import Fitter
import numpy as np

__author__ = 'TimeWz667'
__all__ = ['ABC']


class ABC(Fitter):
    def __init__(self, name_logger="ABC", n_test=100, p_test=0.1):
        Fitter.__init__(self, name_logger, n_test = n_test, p_test=p_test)

    def fit(self, model, **kwargs):
        n_post = kwargs['n_post']
        n_test = self.Parameters['n_test']
        p_test = self.Parameters['p_test']

        self.info('Testing threshold')

        tests = list()

        while len(tests) < n_test:
            p = model.sample_prior()
            li = model.evaluate_likelihood(p)
            if np.isfinite(li):
                tests.append(li)

        eps = np.percentile(tests, (1 - p_test) * 100)

        self.info('Detected epsilon = {:.4g}'.format(eps))


        self.info('Collecting posterior parameters')
        post = list()
        while len(post) < n_post:
            p = model.sample_prior()
            li = model.evaluate_likelihood(p)
            if li < eps:
                continue
            p.LogLikelihood = li
            post.append(p)

        self.info('Completed')

        res = BayesResult(nodes=post, model=model, alg=self)
        res.Benchmarks['Eps'] = eps
        res.Benchmarks['ESS'] = n_post
        res.Benchmarks['Niter'] = n_post
        res.Benchmarks['p_test'] = p_test
        return res

    def is_updatable(self):
        return True

    def update(self, res, **kwargs):
        pass
