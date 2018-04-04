from abc import ABCMeta, abstractmethod
import pandas as pd

__author__ = 'TimeWz667'


class Fitter(metaclass=ABCMeta):
    def __init__(self, model):
        self.Posterior = list()
        self.Model = model

    @abstractmethod
    def fit(self, n_post, **kwargs):
        pass

    @abstractmethod
    def update(self, n_add, **kwargs):
        pass

    def posterior_to_df(self):
        return pd.DataFrame([g.Locus for g in self.Posterior])

    def report_posterior(self):
        df = self.posterior_to_df()
        return df.describe()

    def posterior_to_csv(self, file):
        df = self.posterior_to_df()
        df.to_json(file, orient='records')

    def posterior_to_json(self, file):
        df = self.posterior_to_df()
        df.to_csv(file)