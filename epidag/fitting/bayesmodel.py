import epidag as dag
from abc import ABCMeta, abstractmethod
__author__ = 'TimeWz667'
__all__ = ['BayesianModel']


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


class BayesianModel(metaclass=ABCMeta):
    def __init__(self, bn, root_nodes):
        self.BN = bn
        self.Root = PriorNodeSet(root_nodes)

    def sample_prior(self):
        return self.Root.sample_prior(self.BN)

    def evaluate_prior(self, prior):
        prior.LogPrior = self.Root.evaluate_prior(self.BN, prior)
        return prior.LogPrior

    def get_prior_distributions(self, prior=None):
        prior = prior if prior else self.sample_prior()
        return self.Root.get_prior_distributions(self.BN, prior)

    @property
    @abstractmethod
    def has_exact_likelihood(self):
        pass

    @abstractmethod
    def evaluate_likelihood(self, prior):
        pass
