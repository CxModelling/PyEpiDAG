from abc import ABCMeta, abstractmethod
__author__ = 'TimeWz667'
__all__ = ['CompoundActor', 'SingleActor', 'FrozenSingleActor', 'Sampler']


class SimulationActor(metaclass=ABCMeta):
    def __init__(self, field):
        self.Field = field

    @abstractmethod
    def sample(self, pas=None, **kwargs):
        pass


class CompoundActor(SimulationActor):
    def __init__(self, field, flow, loci):
        SimulationActor.__init__(self, field)
        self.Flow = list(flow)
        self.Loci = loci

    def sample(self, pas=None, **kwargs):
        pas = dict(pas) if pas else dict()
        pas.update(kwargs)
        for loc in self.Flow:
            pas[loc.Name] = loc.sample(pas)
        return self.Loci.sample(pas)

    def sample_with_mediators(self, pas=None, **kwargs):
        pas = dict(pas) if pas else dict()
        pas.update(kwargs)
        res = dict()
        for loc in self.Flow:
            res[loc.Name] = loc.sample(pas)
        res[self.Field] = self.Loci.sample(pas)
        return res

    def __repr__(self):
        return '{} ({})'.format(self.Field, '->'.join(f.Name for f in self.Flow))

    def __str__(self):
        return '->'.join(f.Name for f in self.Flow)


class SingleActor(SimulationActor):
    def __init__(self, field, di):
        SimulationActor.__init__(self, field)
        self.Loci = di

    def sample(self, pas=None, **kwargs):
        pas = dict(pas) if pas else dict()
        pas.update(kwargs)
        return self.Loci.sample(pas)

    def __repr__(self):
        return '{} ({})'.format(self.Field, self.Loci.Func)

    def __str__(self):
        return str(self.Loci.Func)


class FrozenSingleActor(SimulationActor):
    def __init__(self, field, di, pas):
        SimulationActor.__init__(self, field)
        self.Loci = di
        self.Dist = di.get_distribution(pas)

    def sample(self, pas=None, **kwargs):
        return self.Dist.sample()

    def update(self, pas):
        self.Dist = self.Loci.get_distribution(pas)

    def __repr__(self):
        return '{} ({})'.format(self.Field, self.Dist.Name)

    def __str__(self):
        return self.Dist.Name


class Sampler:
    def __init__(self, act: SimulationActor, loc):
        self.Actor = act
        self.Loc = loc

    def __call__(self, **kwargs):
        """
        Sample a value of self.Actor on self.Loc
        :return: a single value
        """
        return self.Actor.sample(self.Loc, **kwargs)

    def sample(self, n=1, **kwargs):
        """
        Sample a list of values
        :param n: integer > 0, size of sampled list
        :return: a list of values
        """
        n = max(n, 1)
        return [self(**kwargs) for _ in range(n)]

    def __repr__(self):
        return 'Actor {} on {}'.format(repr(self.Actor), self.Loc.Nickname)

    __str__ = __repr__
