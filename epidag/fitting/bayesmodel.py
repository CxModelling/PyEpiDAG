from abc import ABCMeta, abstractmethod
from epidag.bayesnet import Chromosome
from epidag.fn import sample, evaluate_nodes

__author__ = 'TimeWz667'
__all__ = ['BayesianModel']


class BayesianModel(metaclass=ABCMeta):
    def __init__(self, bn):
        self.BN = bn
        self.Name = bn.Name

    def sample_prior(self):
        return Chromosome(sample(self.BN))

    def evaluate_prior(self, prior):
        prior.LogPrior = evaluate_nodes(self.BN, prior)
        return prior.LogPrior

    def get_movable_nodes(self):
        p = self.sample_prior()
        res = []
        for root in self.BN.RVRoots:
            if root in p:
                d = self.BN[root].get_distribution(p)
                res.append({'Name': root, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
        return res

    @property
    @abstractmethod
    def has_exact_likelihood(self):
        pass

    @abstractmethod
    def evaluate_likelihood(self, prior):
        pass
