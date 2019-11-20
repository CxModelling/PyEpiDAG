import numpy as np
from scipy.special import expit
from epidag.distribution import parse_distribution
from epidag.data.reg.linear import LinearCombination, Regression


__author__ = ['TimeWz667']
__all__ = ['LogisticRegression', 'PoissonRegression', 'MultinomialLogisticRegression']


class LogisticRegression(Regression):
    def __init__(self, js):
        self.Intercept = js['Intercept']
        self.LC = LinearCombination(js['Regressors'])

    def get_variable_type(self):
        return 'Integer'

    def expectation(self, xs):
        mu = self.LC.predict(xs) + self.Intercept
        return expit(mu)

    def predict(self, xs):
        return np.random.binomial(1, self.expectation(xs))

    def get_sampler(self, xs):
        return parse_distribution('binom(1, p)', {'p': self.expectation(xs)})

    def __str__(self):
        return 'logit(y){}+{}'.format(str(self.LC), self.Intercept)


class PoissonRegression(Regression):
    def __init__(self, js, offset=0):
        self.Intercept = js['Intercept']
        self.LC = LinearCombination(js['Regressors'])
        self.Offset = offset

    def get_variable_type(self):
        return 'Integer'

    def expectation(self, xs):
        mu = self.LC.predict(xs) + self.Intercept
        return np.exp(mu)

    def predict(self, xs):
        return np.random.poisson(1, self.expectation(xs))

    def get_sampler(self, xs):
        return parse_distribution('pois(lam)', {'lam': self.expectation(xs)})

    def __str__(self):
        return 'log(y){}+{}'.format(str(self.LC), self.Intercept)


class MultinomialLogisticRegression(Regression):
    def __init__(self, js):
        self.Intercepts = list(js['Intercepts'])
        self.LCs = [LinearCombination(reg) for reg in js['Regressions']]
        if 'Labels' in js:
            self.Labels = list(js['Labels'])
        else:
            self.Labels = [str(i) for i, _ in enumerate(self.Intercepts)]

    def get_variable_type(self):
        return 'Category'

    def expectation(self, xs):
        ps = [lc.predict(xs) + inc for lc, inc in zip(self.LCs, self.Intercepts)]
        ps = expit(np.array(ps))
        ps = ps / ps.sum()
        return ps

    def predict(self, xs):
        ps = self.expectation(xs)
        return np.random.choice(self.Labels, 1, p=ps)

    def get_sampler(self, xs):
        kv = {l: p for l, p in zip(self.Labels, self.expectation(xs))}
        return parse_distribution('cat(kv)', {'kv': kv})

    def __str__(self):
        return 'mlogit(y){}'.format(str(self.LCs[0]))


if __name__ == '__main__':
    case1 = {'Male': True}
    case2 = {'Male': False}

    reg = [
        {'Name': 'Male', 'Type': 'Boolean', 'Value': 0.5}
    ]

    lr = LogisticRegression({'Intercept': 0, 'Regressors': reg})
    print(lr)
    print(lr.expectation(case1))
    print(sum(lr.get_sampler(case1).sample(1000))/1000)

    print(lr.expectation(case2))
    print(sum(lr.get_sampler(case2).sample(1000))/1000)

    pr = PoissonRegression({'Intercept': 0.1, 'Regressors': reg})
    print(pr)
    print(pr.expectation(case1))
    print(sum(pr.get_sampler(case1).sample(1000))/1000)

    print(pr.expectation(case2))
    print(sum(pr.get_sampler(case2).sample(1000))/1000)

    mlr = MultinomialLogisticRegression({'Intercepts': [0, 2, 2],
                                         'Regressions': [
                                             [{'Name': 'Male', 'Type': 'Boolean', 'Value': 0}],
                                             [{'Name': 'Male', 'Type': 'Boolean', 'Value': 1}],
                                             [{'Name': 'Male', 'Type': 'Boolean', 'Value': -1}],
                                         ],
                                         'Labels': ['A', 'B', 'C']})
    print(mlr)
    from collections import Counter

    print(mlr.expectation(case1))
    print(Counter(mlr.get_sampler(case1).sample(1000)))

    print(mlr.expectation(case2))
    print(Counter(mlr.get_sampler(case2).sample(1000)))
