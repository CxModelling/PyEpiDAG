from epidag.bayesnet import Gene
from .actor import FrozenSingleActor, Sampler
import networkx as nx

__author__ = 'TimeWz667'


class ParameterCore(Gene):
    def __init__(self, nickname, sg, vs, prior):
        Gene.__init__(self, vs, prior)
        self.Nickname = nickname
        self.SG = sg
        self.Parent = None
        self.Actors = dict()
        self.Children = dict()
        self.ChildrenActors = dict()

    @property
    def Group(self):
        return self.SG.Name

    def breed(self, nickname, group):
        """
        Generate an offspring node
        :param nickname: nickname
        :param group: target group of new parameter node
        :return:
        """
        if nickname in self.Children:
            raise ValueError('{} has already existed'.format(nickname))
        chd = self.SG.breed(nickname, group, self)
        self.Children[nickname] = chd
        return chd

    def list_sampler(self):
        for k in self.Actors.keys():
            yield k

        if self.Parent:
            actors = self.Parent.ChildrenActors[self.SG.Name]
            for k in actors:
                yield k

    def get_sampler(self, sampler):
        """
        Get a sampler of a specific variable
        :param sampler: name of the target sampler
        :return:
        """
        try:
            actor = self.Actors[sampler]
        except KeyError:
            try:
                actor = self.Parent.ChildrenActors[self.SG.Name][sampler]
            except AttributeError:
                raise KeyError('No {} found'.format(sampler))
            except KeyError:
                raise KeyError('No {} found'.format(sampler))

        return Sampler(actor, self)

    def get_child(self, name):
        return self.Children[name]

    def find_descendant(self, address):
        """
        Find a descendant node
        :param address: str, a series of names of nodes linked with '@'
        :return: a child node in the address
        """
        sel = self
        names = address.split('@')
        if len(names) < 2:
            return sel
        for name in names[1:]:
            sel = sel.get_child(name)
        return sel

    def impulse(self, imp, shocked=None):
        """
        Do interventions
        :param imp: dict, intervention
        :param shocked: Do not manually input
        """
        imp = dict(imp)
        if shocked is None:
            g = self.SG.SC.BN.DAG
            shocked = set.union(*[set(nx.descendants(g, k)) for k in imp.keys()])

        shocked_locus = [s for s in shocked if s in self.Locus]
        shocked_actors = [k for k, v in self.Actors.items() if k in shocked and isinstance(v, FrozenSingleActor)]

        shocked_hoist = dict()
        for k, v in self.ChildrenActors.items():
            shocked_hoist[k] = [s for s, t in v.items() if s in shocked and isinstance(v, FrozenSingleActor)]

        if imp or shocked_locus or shocked_actors or shocked_hoist:
            self.SG.set_response(imp, shocked_locus, shocked_actors, shocked_hoist, self)

        for v in self.Children.values():
            v.impulse(imp, shocked)

    def reset_sc(self, sc):
        self.SG = sc[self.SG.Name]
        for v in self.Children.values():
            v.reset_sc(self, sc)

    @property
    def DeepLogPrior(self):
        """
        Log prior with that of offsprings
        :return: log prior probability
        """
        return self.LogPrior + sum(v.DeepLogPrior for v in self.Children.values())

    def __iter__(self):
        if self.Parent:
            for v in iter(self.Parent):
                yield v
        for v in self.Locus.items():
            yield v

    def deep_print(self, i=0):
        prefix = '--' * i + ' ' if i else ''
        print('{}{} ({})'.format(prefix, self.Nickname, self))
        for k, chd in self.Children.items():
            chd.deep_print(i + 1)
