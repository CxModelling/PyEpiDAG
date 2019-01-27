import numpy as np
from scipy.special import logsumexp


def ess(lis):
    lis = np.array(lis)
    lis = lis - logsumexp(lis)
    lis = np.exp(lis)
    return len(lis) / ((lis * lis).mean() / np.power(lis.mean(), 2))


def dic(lis, full=False):
    lis = np.array(lis)
    dbar = -2 * lis.mean()
    pd = lis.var()
    if full:
        return {'Dbar': dbar, 'pD': pd, 'DIC': dbar + pd}
    else:
        return dbar + pd
