from epidag.util import resample
from epidag.fitting.alg.fitter import Fitter

__author__ = 'TimeWz667'
__all__ = ['SIR']


class SIR(Fitter):
    DefaultParameters = dict(Fitter.DefaultParameters)

    def __init__(self, bm, **kwargs):
        Fitter.__init__(self, bm, **kwargs)
        self.LogWts = None

    def initialise(self):
        self.Prior.clear()
        self.Posterior.clear()

    def fit(self, **kwargs):
        self.update_parameters(**kwargs)

        self.info('Sampling')
        self.initialise_prior(self['n_population'])

        self.info('Calculating Importance')
        self.LogWts = [p.LogLikelihood for p in self.Prior]

        self.info('Resampling')
        self.Posterior, _ = resample(self.LogWts, self.Prior)

    def update(self, **kwargs):
        self.update_parameters(**kwargs)
        n = self['n_update'] + len(self.Posterior)
        self.Posterior, _ = resample(self.LogWts, self.Prior, new_size=n)
