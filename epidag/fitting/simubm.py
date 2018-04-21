import epidag as dag
from .bayesmodel import BayesianModel

__author__ = 'TimeWz667'
__all__ = ['SimulationBayesianModel']


class SimulationBayesianModel(BayesianModel):
    def __init__(self, sm, data, sim_fn, mea_fun, exact_like=False):
        BayesianModel.__init__(self, sm.BN)
        self.Data = data
        self.SimCore = sm
        self.SimFn = sim_fn
        self.MeasureFn = mea_fun
        self.__exact = bool(exact_like)
        self.Index = 0

    def sample_prior(self):
        p = self.SimCore.generate('Sim{:06d}'.format(self.Index))
        self.Index += 1
        return p

    def evaluate_prior(self, prior):
        prior.LogPrior = dag.evaluate_nodes(self.SimCore.BN, prior)
        return prior.LogPrior

    def get_prior_distributions(self, prior=None):
        prior = prior if prior else self.sample_prior()
        dis = dict()
        bn = self.SimCore.BN
        for k, _ in prior:
            if bn.is_rv(k):
                dis[k] = bn[k].get_distribution(prior)
        return dis

    def evaluate_likelihood(self, prior):
        sim = self.SimFn(prior, self.Data)
        return self.MeasureFn(sim, self.Data)

    @property
    def has_exact_likelihood(self):
        return self.__exact
