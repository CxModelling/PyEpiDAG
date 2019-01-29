from abc import ABCMeta, abstractmethod
__author__ = 'TimeWz667'
__all__ = ['BayesianModel']


class BayesianModel(metaclass=ABCMeta):
    def __init__(self, bn):
        self.BN = bn

    @abstractmethod
    def sample_prior(self):
        pass

    def evaluate_prior(self, prior):
        pass

    def get_movable_nodes(self):
        p = self.sample_prior()
        res = []
        for root in self.BN.RVRoots:
            if root in p:
                d = self.BN[root].get_distribution(p)
                res.append({'Name': root, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
        return res

    def get_prior_distributions(self, prior=None):
        dis = dict()
        for root in self.BN.RVRoots:
            dis[root] = self.BN[root].get_distribution(prior)
        return dis

    @property
    @abstractmethod
    def has_exact_likelihood(self):
        pass

    @abstractmethod
    def evaluate_likelihood(self, prior):
        pass
