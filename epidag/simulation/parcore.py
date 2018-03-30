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

    def breed(self, nickname, group, exo=None):
        """
        Generate an offspring node
        :param nickname: nickname
        :param group: target group of new parameter node
        :param exo: exogenous variables
        :return:
        """
        if nickname in self.Children:
            raise ValueError('{} has already existed'.format(nickname))
        chd = self.SG.breed(nickname, group, self, exo)
        self.Children[nickname] = chd
        return chd

    def duplicate(self, nickname):
        if not self.Parent:
            raise AttributeError('Root node can not be duplicate')
        return self.Parent.bread(nickname, self.Group, exo=self.Locus)

    def detach_from_parent(self, collect_pars=False):
        """
        Remove the reference to it parent
        :param collect_pars: collect the parameters of the parent to itself
        """
        if not self.Parent:
            return
        self.Parent.remove_children(self.Nickname)
        if collect_pars:
            for k, v in self.Parent:
                self[k] = v
        self.Parent = None

    def remove_children(self, k):
        """
        Remove a child ParameterCore
        :param k: the name of the child
        :return: the removed ParameterCore
        """
        try:
            chd = self.Children[k]
            del self.Children[k]
            return chd
        except KeyError:
            pass

    def list_samplers(self):
        li = list(self.Actors.keys())

        if self.Parent:
            actors = self.Parent.ChildrenActors[self.SG.Name]
            li += list(actors.keys())
        return li

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

    def impulse(self, imp):
        """
        Do interventions
        :param imp: dict(node: value) or list(node), intervention
        """
        g = self.SG.SC.BN.DAG
        if isinstance(imp, dict):
            shocked = set.union(*[set(nx.descendants(g, k)) for k in imp.keys()])
            non_imp = [k for k, v in imp.items() if v is None]
            imp = {k: v for k, v in imp.items() if v is not None}
            shocked.difference_update(imp.keys())
            shocked = shocked.union(non_imp)
        elif isinstance(imp, list):
            shocked = set.union(*[set(nx.descendants(g, k)) for k in imp])
            shocked = shocked.union(imp)
            imp = dict()
        else:
            raise AttributeError('imp defined incorrectly')
        # print(shocked)
        self.__set_response(imp, shocked)

    def __set_response(self, imp, shocked):
        shocked_locus = [s for s in shocked if s in self.Locus]
        shocked_actors = [k for k, v in self.Actors.items() if k in shocked and isinstance(v, FrozenSingleActor)]
        shocked_hoist = dict()
        for k, v in self.ChildrenActors.items():
            shocked_hoist[k] = [s for s, t in v.items() if s in shocked and isinstance(v, FrozenSingleActor)]
        # print(shocked_locus, shocked_actors, shocked_hoist)
        self.SG.set_response(imp, shocked_locus, shocked_actors, shocked_hoist, self)

        for v in self.Children.values():
            v.__set_response(imp, shocked)

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

    def __getitem__(self, item):
        try:
            return Gene.__getitem__(self, item)
        except KeyError:
            try:
                return self.Parent[item]
            except AttributeError:
                raise KeyError('{} not found'.format(item))
            except KeyError:
                raise KeyError('{} not found'.format(item))

    def deep_print(self, i=0):
        prefix = '--' * i + ' ' if i else ''
        print('{}{} ({})'.format(prefix, self.Nickname, self))
        for k, chd in self.Children.items():
            chd.deep_print(i + 1)

    def print(self):
        print('{} ({})'.format(self.Nickname, self))

    def clone(self, copy_sc=False):
        if copy_sc:
            sc = self.SG.SC.clone()
            sg = sc.SGs[self.Group]
        else:
            sg = self.SG
        pc_new = sg.generate(self.Nickname, dict(self))
        pc_new.LogPrior = self.LogPrior
        return pc_new
