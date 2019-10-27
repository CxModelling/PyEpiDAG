import numpy as np
from scipy.special import expit
from epidag.bayesnet.distribution import parse_distribution
from epidag.data.reg.linear import LinearCombination, Regression


__author__ = ['TimeWz667']
__all__ = ['LogisticRegression', 'PoissonRegression']


class LogisticRegression(Regression):
    def __init__(self, inc, js):
        self.Intercept = inc
        self.LC = LinearCombination(js)

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
    def __init__(self, inc, js, offset=0):
        self.Intercept = inc
        self.LC = LinearCombination(js)
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
        return 'log(y)~{}+{}'.format(str(self.LC), self.Intercept)


if __name__ == '__main__':
    case1 = {'Male': True}
    case2 = {'Male': False}

    reg = [
        {'Name': 'Male', 'Type': 'Boolean', 'Value': 0.5}
    ]

    lr = LogisticRegression(0, reg)

    print(lr.expectation(case1))
    print(lr.expectation(case2))

    print(sum(lr.get_sampler(case1).sample(1000))/1000)
    print(sum(lr.get_sampler(case2).sample(1000))/1000)

    pr = PoissonRegression(0, reg)
    print(pr.expectation(case1))
    print(pr.expectation(case2))

    print(sum(pr.get_sampler(case1).sample(1000))/1000)
    print(sum(pr.get_sampler(case2).sample(1000))/1000)
