import numpy as np
from epidag.bayesnet.distribution import parse_distribution
from epidag.data.reg.hazard import *
from epidag.data.reg.linear import Regression, LinearCombination

__author__ = 'TimeWz667'
__all__ = ['CoxRegression']


def find_baseline(js):
    tp = js['Type'].lower()
    if tp == 'exp':
        return ExponentialHazard(js['Rate'])
    elif tp == 'weibull':
        return WeibullHazard(js['Lambda'], js['K'])
    elif tp == 'empirical':
        pass # todo EmpiricalHazard(js['Time'], js['CumHaz'])
    raise KeyError('Unknown baseline distribution')


class CoxRegression(Regression):
    def __init__(self, js):
        self.Hazard = find_baseline(js['Baseline'])
        self.LC = LinearCombination(js['Regressors'])

    def get_variable_type(self):
        return 'Double'

    def _rr(self, xs):
        return np.exp(self.LC.predict(xs))

    def expectation(self, xs):
        return HazardDistribution(self.Hazard, self._rr(xs)).mean()

    def predict(self, xs):
        haz = HazardDistribution(self.Hazard, self._rr(xs))
        return haz.sample()

    def get_sampler(self, xs):
        return HazardDistribution(self.Hazard, self._rr(xs))

    def __str__(self):
        return 'surv(y)~{}'.format(str(self.LC))


if __name__ == '__main__':
    case1 = {'Male': True}
    case2 = {'Male': False}

    reg = [
        {'Name': 'Male', 'Type': 'Boolean', 'Value': 0.5}
    ]

    cr = CoxRegression({'Baseline': {'Type': 'exp', 'Rate': 1}, 'Regressors': reg})
    print(cr.expectation(case1))
    print(sum(cr.get_sampler(case1).sample(1000))/1000)

    print(cr.expectation(case2))
    print(sum(cr.get_sampler(case2).sample(1000))/1000)

    cr = CoxRegression({'Baseline': {'Type': 'weibull', 'K': 1, 'Lambda': 0.5},
                        'Regressors': reg})
    print(cr.expectation(case1))
    print(sum(cr.get_sampler(case1).sample(1000)) / 1000)

    print(cr.expectation(case2))
    print(sum(cr.get_sampler(case2).sample(1000)) / 1000)

