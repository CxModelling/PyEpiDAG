from abc import ABCMeta, abstractmethod
__author__ = 'TimeWz667'
__all__ = ['BayesianModel']


class BayesianModel(metaclass=ABCMeta):
    def __init__(self, bn):
        self.BN = bn

    @abstractmethod
    def sample_prior(self):
        pass

    @abstractmethod
    def evaluate_prior(self, prior):
        pass

    @abstractmethod
    def get_prior_distributions(self, prior=None):
        pass

    @property
    @abstractmethod
    def has_exact_likelihood(self):
        pass

    @abstractmethod
    def evaluate_likelihood(self, prior):
        pass
