from abc import ABCMeta, abstractmethod
from epidag.bayesnet.loci import *

__author__ = 'TimeWz667'
__all__ = ['FrozenSingleActor', 'SingleActor', 'CompoundActor', 'Sampler']


class SimulationActor(metaclass=ABCMeta):
    def __init__(self, field, loci, to_read):
        self.Field = field
        self.Loci = loci
        self.ToRead = to_read

    @abstractmethod
    def sample(self, pas=None):
        pass

    def read_upstream(self, pas=None):
        up = dict()
        if self.ToRead and pas is not None:
            for p in self.ToRead:
                try:
                    up[p] = pas[p]
                except KeyError or AttributeError or TypeError:
                    pass
        return up


class FrozenSingleActor(SimulationActor):
    def __init__(self, field, loci, pas):
        SimulationActor.__init__(self, field, loci, pas)
        self.Sampler = None

    def sample(self, pas=None):
        if isinstance(self.Loci, DistributionLoci):
            return self.Sampler.sample()
        else:
            return self.Sampler

    def update(self, pas):
        pas = self.read_upstream(pas)
        if isinstance(self.Loci, DistributionLoci):
            self.Sampler = self.Loci.get_distribution(pas)
        else:
            self.Sampler = self.Loci.render(pas)

    def __repr__(self):
        return '{} ({})'.format(self.Field, str(self))

    def __str__(self):
        return str(self.Loci.Definition)


class SingleActor(SimulationActor):
    def __init__(self, field, loci, pas):
        SimulationActor.__init__(self, field, loci, pas)

    def sample(self, pas=None):
        pas = self.read_upstream(pas)
        return self.Loci.render(pas)

    def __repr__(self):
        return '{} ({})'.format(self.Field, str(self))

    def __str__(self):
        return str(self.Loci.Definition)


class CompoundActor(SimulationActor):
    def __init__(self, field, loci, to_read, to_sample):
        SimulationActor.__init__(self, field, loci, to_read)
        self.Flow = list(to_sample)

    def sample(self, pas=None):
        pas = self.read_upstream(pas)

        for loc in self.Flow:
            pas[loc.Name] = loc.render(pas)
        return self.Loci.render(pas)

    def sample_with_mediators(self, pas=None):
        pas = self.read_upstream(pas)

        res = dict()
        for loc in self.Flow:
            res[loc.Name] = pas[loc.Name] = loc.render(pas)
        res[self.Field] = self.Loci.render(pas)
        return res

    def __repr__(self):
        return '{} ({})'.format(self.Field, str(self))

    def __str__(self):
        return '({})->({})->{}'.format(
            ','.join(f for f in self.ToRead),
            ','.join(f.Name for f in self.Flow),
            self.Loci.Definition)


class Sampler:
    def __init__(self, act: SimulationActor, cms):
        self.Actor = act
        self.Chromosome = None
        self.update(cms)

    def __call__(self):
        """
        Sample a value of self.Actor on self.Loc
        :return: a single value
        """
        return self.Actor.sample(self.Chromosome)

    def update(self, cms=None):
        if cms is not None:
            self.Chromosome = cms

        if isinstance(self.Actor, FrozenSingleActor):
            self.Actor.update(self.Chromosome)

    def sample(self, n=1):
        """
        Sample a list of values
        :param n: integer > 0, size of sampled list
        :return: a list of values
        """
        n = max(n, 1)
        if n is 1:
            return self()
        return [self() for _ in range(n)]

    def __str__(self):
        return 'Actor {} on {}'.format(repr(self.Actor), self.Chromosome.Nickname)

    def __repr__(self):
        return 'Actor( {} on {})'.format(repr(self.Actor), self.Chromosome.Nickname)
