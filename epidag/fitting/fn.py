import numpy as np
import epidag as dag
from epidag.fitting.res import Result



__author__ = 'TimeWz667'
__all__ = ['sample_prior']


def sample_prior(model, n: int = 1000, max_drop: int = 1000):
    prior = []

    n_drop = 0

    while len(prior) < n:
        p = model.sample_prior()
        li = model.evaluate_likelihood(p)
        if np.isfinite(li):
            prior.append(p)
        else:
            n_drop += 1

        if n_drop >= max_drop:
            raise AttributeError('Parameter space might not well-defined')

    return Result(nodes = prior, model = model)
