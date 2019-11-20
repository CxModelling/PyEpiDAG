import numpy as np
from epidag.bayesnet import BayesianNetwork, get_sufficient_nodes

__author__ = 'TimeWz667'
__all__ = ['sample', 'sample_minimally', 'evaluate_nodes']


def sample(bn, cond=None):
    """
    Sample every variables of a Bayesian Network
    :param bn: a Bayesian Network
    :param cond: dict, given variables
    :return:
    """
    g = bn.DAG
    cond = cond if cond else dict()
    if any(nod not in cond for nod in bn.Exo):
        raise ValueError('Exogenous nodes do not fully defined')

    res = dict(cond)

    for nod in bn.Order:
        if nod not in res:
            res[nod] = g.nodes[nod]['loci'].render(res)
    return res


def sample_minimally(bn, included, cond=None, sources=True):
    """
    Sample variables which are minimal requirements of having included
    :param bn: a Bayesian Network
    :param included: iterable, targeted output variables
    :param cond: dict, given variables
    :param sources: True if mediators requested
    :return:
    """
    g = bn.DAG

    cond = cond if cond else dict()
    given = list(cond.keys())

    suf = get_sufficient_nodes(g, included, given)
    suf_exo = [nod for nod in bn.Exo if nod in suf]

    for nod in suf_exo:
        if nod not in cond:
            raise ValueError('Exogenous node {} needed'.format(nod))

    res = dict(cond)

    for nod in bn.Order:
        if nod in suf and nod not in res:
            res[nod] = g.nodes[nod]['loci'].render(res)
    sinks = {k: v for k, v in res.items() if k in included}
    if sources:
        med = {k: v for k, v in res.items() if k not in included}
        return sinks, med
    else:
        return sinks


def evaluate_nodes(bn, pars):
    """
    Evaluate the likelihood of a set of variables
    :param bn: BayesianNetwork, a Bayesian Network
    :type bn: BayesianNetwork
    :param pars: dict, a container of parameters
    :return: the log likelihood of pars
    """
    nodes = bn.DAG.nodes
    lps = np.sum([nodes[k]['loci'].evaluate(pars) for k in pars.keys()])
    return lps


def as_causal_diagram(bn):
    return
