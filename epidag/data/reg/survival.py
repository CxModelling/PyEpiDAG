from epidag.data.reg.hazard import *
from epidag.data.reg.linear import Regression, LinearCombination

__author__ = 'TimeWz667'
__all__ = ['']


def find_baseline(js):
    tp = js['Type'].lower()
    if tp is 'exp':
        return ExponentialHazard(js['Rate'])
    elif tp is 'weibull':
        return WeibullHazard(js['Lambda'], js['K'])
    elif tp is 'empirical':
        pass # todo EmpiricalHazard(js['Time'], js['CumHaz'])
    raise KeyError('Unknown baseline distribution')


class CoxRegression(Regression):
    def __init__(self, js):
        self.Hazard = js['Baseline']
        self.LC = LinearCombination(js['Regressors'])

    def get_variable_type(self):
        return 'Double'

    def _rr(self, xs):
        return self.LC.predict(xs)

    def expectation(self, xs):
        mu = self.LC.predict(xs) + self.Hazard.mean()
        return np.exp(mu)

    def predict(self, xs):
        return np.random.poisson(1, self.expectation(xs))

    def get_sampler(self, xs):
        return parse_distribution('pois(lam)', {'lam': self.expectation(xs)})

    def __str__(self):
        return 'log(y)~{}+{}'.format(str(self.LC), self.Intercept)
