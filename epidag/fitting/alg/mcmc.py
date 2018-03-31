from abc import ABCMeta, abstractmethod
from epidag.fitting.alg.fitter import Fitter
import numpy as np

__author__ = 'TimeWz667'
__all__ = ['MCMC']

"""
Adopted from
    Roberts, Gareth O., and Jeffrey S. Rosenthal. 
    "General state space Markov chains and MCMC algorithms." Probability Surveys 1 (2004): 20-71.
"""


class AbsStepper(metaclass=ABCMeta):
    def __init__(self, name, lo, up):
        self.Name = name
        self.Lower, self.Upper = lo, up
        self.MaxAdaptation = 0.33
        self.InitialAdaptation = 1.0
        self.TargetAcceptance = 0.44
        self.LogStepSize = -1
        self.AcceptanceCount = 0
        self.BatchCount = 0
        self.BatchSize = 50
        self.IterationsSinceAdaption = 0

    @property
    def StepSize(self):
        return np.exp(self.LogStepSize)

    @abstractmethod
    def proposal(self, v, scale):
        pass

    def step(self, bm, gene):
        value = gene[self.Name]
        proposed = self.proposal(value, self.StepSize)
        ng = gene.clone()

        if self.Lower < proposed < self.Upper:
            if gene.LogLikelihood is 0:
                gene.LogLikelihood = bm.evaluate_likelihood(gene)
            ng[self.Name] = proposed
            ng.LogPrior = bm.evaluate_prior(ng)
            ng.LogLikelihood = bm.evaluate_likelihood(ng)
            acc = ng.LogPosterior - gene.LogPosterior
            acc = 100 if acc > 4 else np.exp(acc)

            if acc > np.random.random():
                self.AcceptanceCount += 1
            else:
                ng[self.Name] = gene[self.Name]
                ng.LogPrior = gene.LogPrior
                ng.LogLikelihood = gene.LogLikelihood

        self.IterationsSinceAdaption += 1
        if self.IterationsSinceAdaption >= self.BatchSize:
            self.BatchCount += 1
            adj = min(self.MaxAdaptation, self.InitialAdaptation/np.sqrt(self.BatchCount))
            if self.AcceptanceCount/self.BatchSize > self.TargetAcceptance:
                self.LogStepSize += adj
            else:
                self.LogStepSize -= adj
            self.AcceptanceCount, self.IterationsSinceAdaption = 0, 0
        return ng

    def __str__(self):
        s = 'Stepper ' + self.Name + '{'
        s += 'AcceptanceCount={}, '.format(self.AcceptanceCount)
        s += 'IterationsSinceAdaption={}, '.format(self.IterationsSinceAdaption)
        s += 'LogStepSize={}, '.format(self.LogStepSize)
        s += '}'
        return s


class BinaryStepper(AbsStepper):
    def proposal(self, v, scale):
        return v


class DoubleStepper(AbsStepper):
    def proposal(self, v, scale):
        return v + np.random.normal()*scale


class IntegerStepper(AbsStepper):
    def proposal(self, v, scale):
        return np.round(np.random.normal(v, scale))


class MCMC(Fitter):
    def __init__(self, bm, burn=1000, thin=2):
        Fitter.__init__(self, bm)
        self.BurnIn = burn
        self.Thin = thin
        self.Steppers = list()
        self.Last = None

        for loci, dist in self.Model.sample_distribution().items():
            if dist.Type is 'Double':
                self.Steppers.append(DoubleStepper(loci, dist.Lower, dist.Upper))
            elif dist.Type is 'Integer':
                self.Steppers.append(IntegerStepper(loci, dist.Lower, dist.Upper))
            elif dist.Type is 'Binary':
                self.Steppers.append(DoubleStepper(loci, dist.Lower, dist.Upper))

    def initialise(self):
        self.Posterior.clear()
        self.Last = self.Model.sample_prior()
        self.Last.LogLikelihood = self.Model.evaluate_likelihood(self.Last)

    def fit(self, niter, **kwargs):
        self.initialise()
        ns = 0
        print('Burning in')
        while ns < self.BurnIn:
            for stp in self.Steppers:
                ns += 1
                self.Last = stp.step(self.Model, self.Last)

        print('Gathering posteriori')
        while True:
            for stp in self.Steppers:
                ns += 1
                self.Last = stp.step(self.Model, self.Last)
                if ns % self.Thin is 0:
                    self.Posterior.append(self.Last)
                if len(self.Posterior) >= niter:
                    print('Fitting completed')
                    return

    def update(self, n):
        print('Updating')
        ns = 0
        while True:
            for stp in self.Steppers:
                ns += 1
                self.Last = stp.step(self.Model, self.Last)
                if ns % self.Thin is 0:
                    self.Posterior.append(self.Last)
                if ns >= n:
                    print('Fitting completed')
                    return
