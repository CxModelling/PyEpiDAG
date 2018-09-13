import epidag as dag
from epidag.simulation.fn import *
from epidag.causality.fn import *
from epidag.fitting.fn import *

__author__ = 'TimeWz667'


def sample(bn, cond=None):
    """
    sample every variables of a Bayesian Network
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
            res[nod] = g.nodes[nod]['loci'].sample(res)
    return res


def sample_minimally(bn, included, cond=None, sources=True):
    """
    sample variables which are minimal requirements of having included
    :param bn: a Bayesian Network
    :param included: iterable, targeted output variables
    :param cond: dict, given variables
    :param sources: True if mediators requested
    :return:
    """
    g = bn.DAG

    cond = cond if cond else dict()
    given = list(cond.keys())

    suf = dag.get_sufficient_nodes(g, included, given)
    suf_exo = [nod for nod in bn.Exo if nod in suf]

    for nod in suf_exo:
        if nod not in cond:
            raise ValueError('Exogenous node {} needed'.format(nod))

    res = dict(cond)

    for nod in bn.Order:
        if nod in suf and nod not in res:
            res[nod] = g.nodes[nod]['loci'].sample(res)
    sinks = {k: v for k, v in res.items() if k in included}
    if sources:
        med = {k: v for k, v in res.items() if k not in included}
        return sinks, med
    else:
        return sinks

def as_causal_diagram(bn):
    return
