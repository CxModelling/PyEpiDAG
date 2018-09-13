import epidag as dag
import numpy as np
from .bayesmodel import BayesianModel
from scipy.special import logsumexp

__author__ = 'TimeWz667'
__all__ = ['as_bayesian_model_frame', 'get_data_shadow', 'get_data_hierarchy',
           'PriorNodeSet', 'DataNodeSet', 'DataBayesianModel']


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
    node_list = bn.Order
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


class PriorNodeSet:
    def __init__(self, ns):
        self.Nodes = ns

    def sample_prior(self, bn):
        vs = dag.sample_minimally(bn, included=self.Nodes, sources=False)
        prior = dag.evaluate_nodes(bn, vs)
        return dag.Gene(vs, prior)

    def evaluate_prior(self, bn, gene):
        vs = {k: v for k, v in gene if k in self.Nodes}
        return dag.evaluate_nodes(bn, vs)

    def get_prior_distributions(self, bn, gene):
        dis = dict()
        for k, _ in gene:
            if k in self.Nodes and bn.is_rv(k):
                dis[k] = bn[k].get_distribution(gene)
        return dis

    def __str__(self):
        return "Prior nodes: {}".format(self.Nodes)

    __repr__ = __str__


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
        BayesianModel.__init__(self, bn)
        self.Root = PriorNodeSet(root_nodes)
        self.DataEntries = entries

    def sample_prior(self):
        return self.Root.sample_prior(self.BN)

    def evaluate_prior(self, prior):
        prior.LogPrior = self.Root.evaluate_prior(self.BN, prior)
        return prior.LogPrior

    def get_prior_distributions(self, prior=None):
        prior = prior if prior else self.sample_prior()
        return self.Root.get_prior_distributions(self.BN, prior)

    @property
    def has_exact_likelihood(self):
        return all(not ent.needs_mc for ent in self.DataEntries)

    def evaluate_likelihood(self, prior):
        return np.array([ent.evaluate_likelihood(self.BN, prior) for ent in self.DataEntries]).sum()
