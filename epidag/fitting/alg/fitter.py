from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import logsumexp
from epidag import Monitor
from epidag.fitting import BayesianModel
from epidag.bayesnet import Chromosome


__author__ = 'TimeWz667'


class Fitter(metaclass=ABCMeta):
    DefaultParameters = {
        'n_population': 1000,
        'm_prior_drop': 10,
        'target': 'MLE'
    }

    def __init__(self, model: BayesianModel, **kwargs):
        self.Model = model
        self.Monitor = Monitor(model.BN.Name)
        self.Parameters = dict(self.DefaultParameters)
        self.update_parameters(**kwargs)
        self.Prior = list()
        self.Posterior = list()

    def __getitem__(self, item):
        return self.Parameters[item]

    def update_parameters(self, **kwargs):
        new = {k: v for k, v in kwargs.items() if k in self.Parameters}
        self.Parameters.update(new)

    def renew_parameters(self):
        self.Parameters = dict(self.DefaultParameters)

    def set_log_path(self, filename):
        self.Monitor.set_log_path(filename=filename)

    def info(self, msg):
        self.Monitor.info(msg)

    def error(self, msg):
        self.Monitor.info(msg)

    def initialise_prior(self, n: int=0):
        self.Prior.clear()
        if n <= 0:
            n = self['n_population']
        m = self['m_prior_drop'] * n
        drop = 0

        while len(self.Prior) < n:
            p = self.Model.sample_prior()
            li = self.Model.evaluate_likelihood(p)
            if np.isfinite(li):
                self.Prior.append(p)
            else:
                drop += 1
                if drop >= m:
                    self.error('Too many infinite likelihood in the space')
                    raise AttributeError('Parameter space might not well-defined')

        pr_drop = drop/(len(self.Prior) + drop) * 100
        self.info('Prior parameters generated with {:.1f}% dropouts'.format(pr_drop))

    def prior_to_df(self):
        return Chromosome.to_data_frame(self.Prior)

    def summarise_prior(self):
        print(self.prior_to_df().describe())

    def prior_to_json(self, file):
        df = self.posterior_to_df()
        df.to_json(file, orient='records')

    def prior_to_csv(self, file):
        df = self.posterior_to_df()
        df.to_csv(file)

    def posterior_to_df(self):
        return Chromosome.to_data_frame(self.Posterior)

    def summarise_posterior(self):
        print(self.posterior_to_df().describe())

    def posterior_to_json(self, file):
        df = self.posterior_to_df()
        df.to_json(file, orient='records')

    def posterior_to_csv(self, file):
        df = self.posterior_to_df()
        df.to_csv(file)

    @abstractmethod
    def fit(self, **kwargs):
        pass

    def update(self, **kwargs):
        raise AttributeError('The algorithm does not support update scheme')


class EvolutionaryFitter(Fitter, metaclass=ABCMeta):
    DefaultParameters = dict(Fitter.DefaultParameters)
    DefaultParameters['max_generation'] = 30
    DefaultParameters['n_update'] = 10

    def __init__(self, model, **kwargs):
        Fitter.__init__(self, model, **kwargs)
        self.BestFit = self.Model.sample_prior()
        self.Generation = 0
        self.Stay = 0

    def find_best(self):
        if self.Parameters['target'] == 'MLE':
            key = lambda x: x.LogLikelihood
            x0 = self.BestFit.LogLikelihood
        else:
            key = lambda x: x.LogPosterior
            x0 = self.BestFit.LogPosterior

        if self.Posterior:
            self.BestFit = max(self.Posterior, key=key)
        else:
            self.error('No parameters found')

        if self.Parameters['target'] == 'MLE':
            x1 = self.BestFit.LogLikelihood
        else:
            x1 = self.BestFit.LogPosterior

        if x1 == x0:
            self.Stay += 1
        else:
            self.Stay = 0

    def fit(self, **kwargs):
        if kwargs:
            self.update_parameters(**kwargs)
            self.info('Parameters updated')

        mg = self.Parameters['max_generation']
        self.genesis()
        for i in range(mg):
            self.Generation += 1
            self.step()
            self.find_best()
            if self.termination():
                self.info('Termination criteria reached')
                break
            if self.Generation <= mg:
                self.info('Max generation reached')
                break

        self.info('Fitting completed')

    @abstractmethod
    def step(self):
        pass

    def genesis(self):
        self.Generation = 0
        self.Stay = 0
        self.info('Genesis')
        self.initialise_prior()
        self.Posterior = list(self.Prior)
        self.find_best()

    @abstractmethod
    def termination(self):
        pass

    def keep_record(self):
        if self.Parameters['target'] == 'MLE':
            ma = self.BestFit.LogLikelihood
            vs = [p.LogLikelihood for p in self.Posterior]
        else:
            ma = self.BestFit.LogPosterior
            vs = [p.LogPosterior for p in self.Posterior]

        self.Monitor.keep(
            Max=ma,
            Mean=logsumexp(vs),
            Stay=self.Stay
        )
        self.Monitor.step(self.Generation)

    def update(self, **kwargs):
        self.update_parameters(**kwargs)
        self.info('Parameters updated')

        n_update = self.Parameters['n_update']
        gen = self.Generation + n_update

        for i in range(n_update):
            self.Generation += 1
            self.step()
            self.find_best()

            if self.Generation <= gen:
                self.info('Max generation reached')
                break
        self.info('Update completed')
