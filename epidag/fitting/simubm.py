import epidag as dag
import numpy as np
from epidag.fitting.bayesmodel import BayesianModel


__author__ = 'TimeWz667'




class SimulationBayesianModel(BayesianModel):
    def __init__(self, sm, sim_fn, mea_fun, exact_like=False):
        BayesianModel.__init__(sm.BN, [])
        self.SimCore = sm
        self.SimFn = sim_fn
        self.MeasureFn = mea_fun
        self.__exact = bool(exact_like)

    def evaluate_likelihood(self, prior):
        return self.__exact

    @property
    def has_exact_likelihood(self):
        return self.__exact