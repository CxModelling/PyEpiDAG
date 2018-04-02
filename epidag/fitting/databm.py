import epidag as dag
import numpy as np
from epidag.fitting.bayesmodel import BayesianModel
from scipy.special import logsumexp

__author__ = 'TimeWz667'
__all__ = ['as_data_model', 'DataBayesianModel']


def as_bayesian_model_frame(data, exo=None, datum_name='entries'):
    res = dict()
    if exo:
        res['exogenous'] = dict(exo)

    if datum_name in data:
        res['entries'] = data[datum_name]
    elif isinstance(data, dict):
        res['entries'] = [data]
    elif isinstance(data, list):
        res['entries'] = list(data)
    else:
        raise TypeError('Unknown data structure')

    return res


def get_data_shadow(data, bn):
    node_list = bn.OrderedNodes
    res = dict()
    try:
        res['exogenous'] = [exo for exo in data['exogenous'].keys() if exo in node_list]
    except KeyError:
        res['exogenous'] = list()
    res['entries'] = [[k for k in entry.keys() if k in node_list] for entry in data['entries']]
    return res


def get_data_hierarchy(data, bn, latent=None):
    hierarchy = dict()

    hierarchy['root'] = ['entries'] + data['exogenous']

    nodes = set.union(*[set(ent) for ent in data['entries']])

    if latent:
        nodes = nodes.union([lat for lat in latent if lat in bn.OrderedNodes])
    hierarchy['entries'] = nodes

    return hierarchy


def as_data_model(bn, data, exo=None, latent=None, datum_name='entries'):
    data_reformed = as_bayesian_model_frame(data, exo=exo, datum_name=datum_name)
    data_shadow = get_data_shadow(data_reformed, bn)
    data_hie = get_data_hierarchy(data_shadow, bn, latent=latent)

    ng = dag.form_hierarchy(bn, data_hie)

    root_nodes = bn.sort(ng.Nodes)
    # leaf_nodes = bn.sort(list(ng.Children)[0].Nodes)

    leaves = list()

    for datum, nodes in zip(data_reformed['entries'], data_shadow['entries']):
        min_nodes = dag.get_minimal_nodes(bn.DAG, nodes, root_nodes)
        diff = min_nodes - datum.keys()

        if any(bn.is_rv(d) for d in diff):
            need_mc = True
        else:
            need_mc = False
        min_nodes = bn.sort(min_nodes)
        leaves.append(DataNodeSet(datum, min_nodes, need_mc))

    return DataBayesianModel(bn, root_nodes, leaves)


class DataNodeSet:
    def __init__(self, datum, ns, mc=False):
        self.Datum = {k: v for k, v in datum.items() if k in ns}
        self.Nodes = ns
        self.__MC = mc

    @property
    def needs_mc(self):
        return self.__MC

    def evaluate_likelihood(self, bn, prior):
        if not self.Datum:
            return 0

        fixed = dict(prior.Locus)
        fixed.update(self.Datum)
        if self.needs_mc:
            lis = []
            for _ in range(DataBayesianModel.DefaultMC):
                li = 0
                iteration, src = dag.sample_minimally(bn, self.Nodes, cond=fixed, sources=True)
                iteration.update(src)
                li += np.sum([bn[k].evaluate(iteration) for k in self.Nodes])
                lis.append(li)
            return logsumexp(lis) - np.log(DataBayesianModel.DefaultMC)

        else:
            fixed, src = dag.sample_minimally(bn, self.Nodes, cond=fixed, sources=True)
            fixed.update(src)
            return np.sum([bn[k].evaluate(fixed) for k in self.Nodes])

    def __str__(self):
        return "{} => {}".format(self.Nodes, self.Datum)

    __repr__ = __str__


class DataBayesianModel(BayesianModel):
    DefaultMC = 100

    def __init__(self, bn, root_nodes, entries):
        BayesianModel.__init__(self, bn, root_nodes)
        self.DataEntries = entries

    @property
    def has_exact_likelihood(self):
        return all(not ent.needs_mc for ent in self.DataEntries)

    def evaluate_likelihood(self, prior):
        return np.array([ent.evaluate_likelihood(self.BN, prior) for ent in self.DataEntries]).sum()
