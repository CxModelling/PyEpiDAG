from abc import ABCMeta, abstractmethod
from epidag.bayesnet.loci import DistributionLoci

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
        if self.ToRead and pas:
            for p in self.ToRead:
                try:
                    up[p] = pas[p]
                except KeyError or AttributeError or TypeError:
                    pass
        return up


class FrozenSingleActor(SimulationActor):
    def __init__(self, field, loci, pas):
        SimulationActor.__init__(self, field, loci, list(pas.keys()))
        self.Sampler = None
        self.update(pas)

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

    def sample_with_mediators(self, pas=None, **kwargs):
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
    def __init__(self, act: SimulationActor, chr):
        self.Actor = act
        self.Chromosome = chr

    def __call__(self):
        """
        Sample a value of self.Actor on self.Loc
        :return: a single value
        """
        return self.Actor.sample(self.Chromosome)

    def update(self, chr=None):
        if chr:
            self.Chromosome = chr

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

    def __repr__(self):
        return 'Actor {} on {}'.format(repr(self.Actor), self.Chromosome.Nickname)

    __str__ = __repr__


if __name__ == '__main__':
    from epidag.bayesnet.loci import *

    f1 = FrozenSingleActor('A', DistributionLoci('A', 'k(a)'), {'a': 1})
    print(f1.sample())

    f1.update({'a': 3})
    print(f1.sample())

    f2 = FrozenSingleActor('B', FunctionLoci('B', 'b+4'), {'b': 1})
    print(f2.sample())

    f2.update({'b': 3})
    print(f2.sample())

    print(f2.Sampler)

    s1 = SingleActor('C', DistributionLoci('C', 'k(c)'), ['c'])
    try:
        print(s1.sample())
    except KeyError:
        print('error')

    print(s1)
    print(s1.sample({'c': 3}))


    c1 = CompoundActor('D', DistributionLoci('D', 'k(d)'),
                       ['b'],
                       [
                           DistributionLoci('a', 'k(1)'),
                           FunctionLoci('c', 'a+b'),
                           FunctionLoci('d', 'c+1')
                       ])
    print(c1)
    print(c1.sample({'b': 4}))

    c2 = CompoundActor('D', FunctionLoci('D', 'pow(d, 2)'),
                       ['b'],
                       [
                           DistributionLoci('a', 'k(1)'),
                           FunctionLoci('c', 'a+b'),
                           FunctionLoci('d', 'c+1')
                       ])
    print(c2)
    print(c2.sample({'b': 4}))


    sam1 = Sampler(c2, {'b': 4})
    print(sam1.sample())
    print(sam1.sample(3))
