from abc import ABCMeta, abstractmethod
import logging
from collections import OrderedDict
import pandas as pd
from ..misc import dic, ess

__author__ = 'TimeWz667'


class Fitter(metaclass=ABCMeta):
    def __init__(self, model):
        self.Model = model
        self.Logger = logging.getLogger()

    def log_on(self, log=None):
        if isinstance(log, logging.Logger):
            self.Logger = log
        elif isinstance(log, str):
            self.Logger = logging.getLogger(log)
            self.Logger.setLevel(logging.INFO)
            self.Logger.addHandler(logging.StreamHandler())
        else:
            self.Logger = logging.getLogger()
            self.Logger.setLevel(logging.INFO)
            self.Logger.addHandler(logging.StreamHandler())

    def log_off(self):
        self.Logger = False

    def info(self, msg):
        if not self.Logger:
            return
        self.Logger.info(msg)

    def error(self, msg):
        if not self.Logger:
            return
        self.Logger.error(msg)

    @abstractmethod
    def fit(self, n_post, **kwargs):
        pass

    @abstractmethod
    def update(self, n_add, **kwargs):
        pass


class BayesianFitter(Fitter, metaclass=ABCMeta):
    def __init__(self, model):
        Fitter.__init__(self, model)
        self.Prior = list()
        self.Posterior = list()

    def prior_to_df(self):
        return pd.DataFrame([g.Locus for g in self.Prior])

    def summarise_prior(self):
        print(self.prior_to_df().describe())
        res = OrderedDict()
        res['N'] = len(self.Prior)
        print(pd.Series(res))

    def prior_to_json(self, file):
        df = self.posterior_to_df()
        df.to_json(file, orient='records')

    def prior_to_csv(self, file):
        df = self.posterior_to_df()
        df.to_csv(file)

    def posterior_to_df(self):
        return pd.DataFrame([g.Locus for g in self.Posterior])

    def summarise_posterior(self):
        print(self.posterior_to_df().describe())
        print()
        res = OrderedDict()
        res['N'] = len(self.Posterior)
        lis = [p.LogLikelihood for p in self.Posterior]
        res['ESS'] = ess(lis)
        res.update(dic(lis, full=True))
        print(pd.Series(res))

    def posterior_to_json(self, file):
        df = self.posterior_to_df()
        df.to_json(file, orient='records')

    def posterior_to_csv(self, file):
        df = self.posterior_to_df()
        df.to_csv(file)


class FrequentistFitter(Fitter, metaclass=ABCMeta):
    def __init__(self, model):
        Fitter.__init__(self, model)
        self.BestFit = {}


