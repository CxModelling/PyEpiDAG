from abc import ABCMeta, abstractmethod
__author__ = 'TimeWz667'
__all__ = ['CompoundActor', 'SingleActor', 'FrozenSingleActor', 'FrozenSingleFunctionActor', 'Sampler']


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
        self.Parents = set.union(*[flo.Parents for flo in self.Flow])
        self.Parents = self.Parents.union(self.Loci.Parents)

    def sample(self, pas=None, **kwargs):
        parents = dict()
        for p in self.Parents:
            try:
                parents[p] = pas[p]
            except KeyError:
                parents[p] = kwargs[p]

        for loc in self.Flow:
            parents[loc.Name] = loc.sample(parents)
        return self.Loci.sample(parents)

    def sample_with_mediators(self, pas=None, **kwargs):
        parents = dict()
        for p in self.Parents:
            try:
                parents[p] = pas[p]
            except KeyError:
                try:
                    parents[p] = kwargs[p]
                except KeyError:
                    pass

        res = dict()
        for loc in self.Flow:
            res[loc.Name] = parents[loc.Name] = loc.sample(parents)
        res[self.Field] = self.Loci.sample(parents)
        return res

    def __repr__(self):
        return '{} ({})'.format(self.Field, '->'.join(f.Name for f in self.Flow))

    def __str__(self):
        return '->'.join(f.Name for f in self.Flow)


class SingleActor(SimulationActor):
    def __init__(self, field, di):
        SimulationActor.__init__(self, field)
        self.Loci = di
        self.Parents = set(self.Loci.Parents)

    def sample(self, pas=None, **kwargs):
        parents = dict()
        for p in self.Parents:
            try:
                parents[p] = pas[p]
            except KeyError:
                try:
                    parents[p] = kwargs[p]
                except KeyError as e:
                    raise e
        return self.Loci.sample(parents)

    def __repr__(self):
        return '{} ({})'.format(self.Field, self.Loci.Func)

    def __str__(self):
        return str(self.Loci.Func)


class FrozenSingleFunctionActor(SimulationActor):
    def __init__(self, field, di, pas):
        SimulationActor.__init__(self, field)
        self.Loci = di
        self.Pars = pas

    def sample(self, pas=None, **kwargs):
        return self.Loci.sample(self.Pars)

    def update(self, pas):
        self.Pars = pas

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
        return '{} ({})'.format(self.Field, repr(self.Dist))

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
