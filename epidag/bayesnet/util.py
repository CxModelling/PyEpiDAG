import numpy as np
from numpy.random import uniform
from scipy.misc import logsumexp

__author__ = 'TimeWz667'
__all__ = ['ScriptException', 'Sampler', 'resample']


class ScriptException(Exception):
    def __init__(self, err):
        self.Err = err

    def __repr__(self):
        return self.Err

class Sampler(object):
    """
    A sampler to sample random integer with weight

    Attributes:
        _q (float array): The weight.
    """
    def __init__(self, q):
        """
        Args:
            q (float array): The weight.
        """
        self._q = np.asarray(q)
        self.Q = np.cumsum(q)
        self.Total = self._q.sum()

    def __repr__(self):
        return "DiscreteRV with {n} elements".format(n=self._q.size)

    def __str__(self):
        return self.__repr__()

    def __call__(self, k=1):
        """
        Args:
            k (int): number of required random integer
        Return:
            int: a random integer or an integer array  
        """
        r = self.Q.searchsorted(uniform(0, self.Total, size=k))
        if k is 1:
            return r[0]
        else:
            return r


def resample(logwts, hs, pars=None):
    size = len(logwts)
    wts = np.array(logwts)
    lse = logsumexp(wts)
    wts -= lse
    sam = Sampler(np.exp(wts))
    sel = sam(size)
    if pars:
        return [hs[i] for i in sel], [pars[i] for i in sel], lse - np.log(size)
    else:
        return [hs[i] for i in sel], lse - np.log(size)


if __name__ == '__main__':
    from collections import Counter
    sam1 = Sampler([1, 2, 3, 4])
    ct1 = Counter(sam1(10000))
    print(ct1)

    sam1 = Sampler([1, 0, 1])
    ct1 = Counter(sam1(10000))
    print(ct1)
