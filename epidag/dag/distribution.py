import scipy.stats as sts
from scipy.interpolate import interp1d
import numpy as np
import numpy.random as rd
from abc import ABCMeta, abstractmethod
from factory import getWorkshop
import factory.arguments as vld


__author__ = 'TimeWz667'
__all__ = ['AbsDistribution', 'SpDouble', 'SpInteger', 'DistributionCentre', 'parse_distribution']


class AbsDistribution(metaclass=ABCMeta):
    def __init__(self, name):
        self.Name = name
        self.json = None

    @property
    @abstractmethod
    def Interval(self):
        pass

    @property
    def Upper(self):
        return self.Interval[1]

    @property
    def Lower(self):
        return self.Interval[0]

    @property
    @abstractmethod
    def Type(self):
        pass

    @abstractmethod
    def sample(self, n=1):
        pass

    @abstractmethod
    def logpdf(self, v):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def std(self):
        pass

    def to_json(self):
        return self.json

    def __repr__(self):
        return self.Name

    __str__ = __repr__


class SpDouble(AbsDistribution):
    def __init__(self, name, dist):
        AbsDistribution.__init__(self, name)
        self.Dist = dist

    def sample(self, n=1):
        if n is 1:
            return self.Dist.rvs()
        return self.Dist.rvs(n)

    def Interval(self):
        return self.Dist.interval(1)

    def Type(self):
        return 'Double'

    def logpdf(self, v):
        return self.Dist.logpdf(v)

    def mean(self):
        return self.Dist.mean()

    def std(self):
        return self.Dist.std()

    def __repr__(self):
        return self.Name

    __str__ = __repr__


class SpInteger(AbsDistribution):
    def __init__(self, name, dist):
        AbsDistribution.__init__(self, name)
        self.Dist = dist

    def sample(self, n=1):
        if n is 1:
            return round(self.Dist.rvs())
        return np.round(self.Dist.rvs(n))

    def Interval(self):
        inter = self.Dist.interval(1)
        return inter[0]+1, inter[1]

    def Type(self):
        return 'Integer'

    def logpdf(self, v):
        return self.Dist.logpmf(v)

    def mean(self):
        return self.Dist.mean()

    def std(self):
        return self.Dist.std()

    def __repr__(self):
        return self.Name

    __str__ = __repr__


class Const(AbsDistribution):
    def __init__(self, name, k):
        """
        Distribution function always draws a constant value
        :param k: value
        """
        AbsDistribution.__init__(self, name)
        self.K = k

    @property
    def Interval(self):
        return [self.K, self.K]

    @property
    def Type(self):
        return type(self.K)

    def sample(self, n=1):
        if n > 1:
            return np.array([self.K] * n)
        return self.K

    def logpdf(self, v):
        return 0 if v == self.K else float('inf')

    def mean(self):
        return self.K

    def std(self):
        return 0


class CategoricalRV(AbsDistribution):
    """
    Generate Categorical data with respect to a specific probability distribution.
    """

    def __init__(self, name, kv):
        """

        :param kv: a dictionary with keys of category names and values of probabilities.
        """
        AbsDistribution.__init__(self, name)
        self.kv = kv
        self.cat = [k for k in kv.keys()]
        self.p = np.array(list(kv.values()))
        self.p = self.p / self.p.sum()

    @property
    def Interval(self):
        return None, None

    @property
    def Type(self):
        return 'Category'

    def logpdf(self, v):
        return np.array([x*np.log(self.kv[k]) for k, x in v.items()]).sum()

    def sample(self, n=1):
        return rd.choice(self.cat, n, p=self.p)

    def mean(self):
        return 0

    def std(self):
        return 0


class EmpiricalRV(AbsDistribution):
    def __init__(self, name, x, y):
        x, y = np.array(x), np.array(y)
        self.X = x
        self.Y = y
        self.__int = (x.min(), x.max())
        AbsDistribution.__init__(self, name)
        self.Fn = interp1d(y.cumsum()/y.sum(), x, bounds_error=False, fill_value=(x.min(), x.max()))
        self.Logpdf = interp1d(x, y, bounds_error=False, fill_value=0)

    @property
    def Interval(self):
        return self.__int

    @property
    def Type(self):
        return 'Double'

    def logpdf(self, v):
        return self.Logpdf(v)

    def sample(self, n=1):
        return self.Fn(rd.random(n))

    def mean(self):
        return np.mean(self.X*self.Fn(self.X))

    def std(self):
        return np.mean(self.X*self.X*self.Fn(self.X))


DistributionCentre = getWorkshop('Distributions')

DistributionCentre.register('k', Const, [vld.Float('k')])


def d_gamma(name, shape, rate):
    return SpDouble(name, sts.gamma(a=shape, scale=1/rate))


DistributionCentre.register('gamma', d_gamma, [vld.PositiveFloat('shape', default=1.0),
                                               vld.PositiveFloat('rate', default=1.0)])


def d_exp(name, rate):
    return SpDouble(name, sts.expon(scale=1/rate))


DistributionCentre.register('exp', d_exp, [vld.PositiveFloat('rate', default=1.0)])


def d_lnorm(name, meanlog, sdlog):
    return SpDouble(name, sts.lognorm(s=meanlog, scale=np.exp(sdlog)))


DistributionCentre.register('lnorm', d_lnorm, [vld.Float('meanlog', default=0),
                                               vld.PositiveFloat('sdlog', default=1.0)])


def d_unif(name, min, max):
    return SpDouble(name, sts.uniform(min, max-min))


DistributionCentre.register('unif', d_unif, [vld.Float('min', default=0),
                                             vld.Float('max', default=1.0)])


def d_chi2(name, df):
    return SpDouble(name, sts.chi2(df))


DistributionCentre.register('chisq', d_chi2, [vld.PositiveFloat('df', default=1.0)])


def d_beta(name, shape1, shape2):
    return SpDouble(name, sts.beta(shape1, shape2))


DistributionCentre.register('beta', d_beta, [vld.PositiveFloat('shape1', default=1.0),
                                             vld.PositiveFloat('shape2', default=1.0)])


def d_invgamma(name, a, rate):
    return SpDouble(name, sts.invgamma(a=a, scale=1/rate))


DistributionCentre.register('invgamma', d_invgamma, [vld.PositiveFloat('a', default=1.0),
                                                     vld.PositiveFloat('rate', default=1.0)])


def d_norm(name, mean, sd):
    return SpDouble(name, sts.norm(loc=mean, scale=sd))


DistributionCentre.register('norm', d_norm, [vld.PositiveFloat('mean', default=0),
                                             vld.PositiveFloat('sd', default=1.0)])


def d_triangle(name, a, m, b):
    x = [a, m, b]
    x.sort()
    a, m, b = x
    return SpDouble(name, sts.triang(loc=a, scale=b - a, c=(m - a) / (b - a)))


DistributionCentre.register('triangle', d_triangle, [vld.PositiveFloat('a', default=0),
                                                     vld.PositiveFloat('m', default=0.5),
                                                     vld.PositiveFloat('b', default=1.0)])


def d_binom(name, size, prob):
    return SpInteger(name, sts.binom(n=size, p=prob))


DistributionCentre.register('binom', d_binom, [vld.PositiveInteger('size', default=1),
                                               vld.Prob('prob', default=0.5)])


DistributionCentre.register('cat', CategoricalRV, [vld.ProbTab('kv')])


def parse_distribution(name, di=None):
    return DistributionCentre.parse(name, di)


if __name__ == '__main__':
    dists = [
        'exp(0.01)',
        'gamma(0.01, 1)',
        'lnorm(0.5, 1)',
        'k(1)',
        'unif(0, 1)',
        'chisq(20)',
        'triangle(2, 3, 5)',
        'binom(size=4, prob=0.5)'
    ]
    dists = [parse_distribution(di) for di in dists]
    for di in dists:
        print(di.to_json())
        print(di.mean())

    dist_cat = parse_distribution('cat({"M": 3411,"O": 3502,"Y": 52})')
    from collections import Counter
    print(dist_cat.to_json())
    print(Counter(dist_cat.sample(10000)))
