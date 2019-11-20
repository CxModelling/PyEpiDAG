import numpy as np
import numpy.random as rd
from scipy.special import gamma
from scipy.interpolate import interp1d
from abc import ABCMeta, abstractmethod
from epidag.distribution import AbsDistribution


__all__ = ['Hazard', 'ExponentialHazard', 'WeibullHazard', 'EmpiricalHazard',
           'HazardDistribution', 'ZeroInflatedHazardDistribution']


class Hazard(metaclass=ABCMeta):
    @abstractmethod
    def cum_hazard(self, t):
        pass

    @abstractmethod
    def inv_cum_hazard(self, h):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def std(self):
        pass


class ExponentialHazard(Hazard):
    def __init__(self, rate):
        self.Rate = rate

    def cum_hazard(self, t):
        return t * self.Rate

    def inv_cum_hazard(self, h):
        return h / self.Rate

    def mean(self):
        return 1 / self.Rate

    def std(self):
        return 1 / self.Rate


class WeibullHazard(Hazard):
    def __init__(self, lam, k):
        self.Lambda = lam
        self.K = k

    def cum_hazard(self, t):
        return pow(self.Lambda * t, self.K)

    def inv_cum_hazard(self, h):
        return pow(h, 1/self.K) / self.Lambda

    def mean(self):
        return gamma(1+1/self.K) * self.Lambda

    def std(self):
        return np.sqrt(gamma(1+2/self.K) - gamma(1+1/self.K)**2) * self.Lambda


class EmpiricalHazard(Hazard):
    def __init__(self, times, cum_haz):
        self.MaxTime = max(times)
        self.MaxHaz = max(cum_haz)
        self.FnCumHaz = interp1d(times, cum_haz, fill_value=[0, self.MaxHaz])
        self.FnInvCumHaz = interp1d(cum_haz, times, fill_value=[0, self.MaxTime])

    def cum_hazard(self, t):
        return self.FnCumHaz(t)

    def inv_cum_hazard(self, h):
        return self.FnInvCumHaz(h)

    def mean(self):
        return None  # todo

    def std(self):
        return None  # todo


class HazardDistribution(AbsDistribution):
    def __init__(self, haz: Hazard, rr):
        AbsDistribution.__init__(self)
        self.Hazard = haz
        self.RiskRatio = rr

    @property
    def Interval(self):
        return [0, float('inf')]

    @property
    def Type(self):
        return 'Double'

    def sample(self, n=1, **kwargs):
        ss = rd.exponential(1/self.RiskRatio, n)
        if n <= 1:
            return self.Hazard.inv_cum_hazard(ss[0])
        else:
            return [self.Hazard.inv_cum_hazard(s) for s in ss]

    def logpdf(self, v):
        s1 = np.expm1(-self.Hazard.cum_hazard(v))
        s2 = np.expm1(-self.Hazard.cum_hazard(v - 1E-5))
        return np.log((s1 - s2)/1E-5)

    def mean(self):
        return 1 / self.Hazard.mean() / self.RiskRatio

    def std(self):
        return pow(self.RiskRatio, -2) / self.Hazard.std()


class ZeroInflatedHazardDistribution(HazardDistribution):
    def __init__(self, pr, haz: Hazard, rr):
        HazardDistribution.__init__(self, haz, rr)
        self.Prob = pr

    def logpdf(self, v):
        if v <= 0:
            return np.log(self.Prob)
        else:
            return np.log(1-self.Prob) + HazardDistribution.logpdf(self, v)

    def sample(self, n=1, **kwargs):
        ps = rd.random(n) < self.Prob
        ts = HazardDistribution.sample(self, n)
        if n <= 1:
            return ts if ps[0] < 1 else 0
        else:
            ts = np.array(ts)
            ts[ps > 0] = 0
            return ts

    def mean(self):
        return HazardDistribution.mean(self) * (1-self.Prob)
