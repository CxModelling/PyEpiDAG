import numpy as np
from epidag.util import resample
from epidag.fitting import BayesResult
from epidag.fitting.alg.fitter import Fitter
from epidag.fitting.misc import ess, dic

__author__ = 'TimeWz667'
__all__ = ['SIR']


class SIR(Fitter):
    def __init__(self, name_logger="SIR", n_test=100, p_test=0.1):
        Fitter.__init__(self, name_logger, n_test = n_test, p_test=p_test)

    def fit(self, model, **kwargs):
        n_post = kwargs['n_post']

        self.info('Sampling')
        prior = list()
        lis = list()
        while len(prior) < n_post:
            p = model.sample_prior()
            li = model.evaluate_likelihood(p)
            if np.isfinite(li):
                p.LogLikelihood = li
                prior.append(p)
                lis.append(li)

        self.info('Importance')

        post, _ = resample(lis, prior)

        self.info('Resampling')
        res = BayesResult(nodes=post, model=model, alg=self)

        res.Benchmarks['ESS'] = ess(lis)
        res.Benchmarks['DIC'] = dic(lis)
        res.Benchmarks['Niter'] = n_post
        return res

    def is_updatable(self):
        return False


if __name__ == '__main__':
    from epidag.fitting import BayesianModel
    from epidag.bayesnet import bayes_net_from_script
    scr = '''
    PCore test {
        prob ~ beta(1, 1)
        x ~ binom(n, prob)    
    }
    '''

    class BinBeta(BayesianModel):
        def __init__(self, bn, data):
            BayesianModel.__init__(self, bn, pars=['prob'])
            self.Data = data

        @property
        def has_exact_likelihood(self):
            return True

        def evaluate_distance(self, pars):
            return - self.evaluate_likelihood(pars)

        def evaluate_likelihood(self, pars):
            li = 0
            pars = dict(pars)
            for d in self.Data:
                pars.update(d)
                li += self.BN['x'].evaluate(pars)
            return li

    bn = bayes_net_from_script(scr)

    data = [
        {'id': 1, 'n': 10, 'x': 4},
        {'id': 2, 'n': 20, 'x': 7}
    ]

    model = BinBeta(bn, data)

    alg = SIR()

    res = alg.fit(model, n_post=1000)

    print(res.summarise())
    print(res.Benchmarks)
