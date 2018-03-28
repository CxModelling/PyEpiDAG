from .actor import CompoundActor, FrozenSingleActor, SingleActor
from .parcore import ParameterCore
import networkx as nx

__author__ = 'TimeWz667'


class SimulationGroup:
    def __init__(self, name, fixed, random, actors, pas):
        self.Name = name
        self.SC = None
        self.Listening = list(pas)
        self.BeFixed = set(fixed)
        self.BeRandom = set(random)
        self.BeActors = set(actors)
        self.FixedChain = None
        self.Children = list()

    def set_simulation_core(self, sc):
        self.SC = sc
        bn = sc.BN
        self.FixedChain = [bn[node] for node in bn.sort(self.BeFixed)]

    def __form_actor(self, bn, g, act, pas):
        pa = set(g.predecessors(act))
        if pa.intersection(self.BeRandom):
            flow = set.intersection(nx.ancestors(g, act), self.BeRandom)
            flow = bn.sort(flow)
            flow = [bn[nod] for nod in flow]
            return CompoundActor(act, flow, bn[act])
        elif pa.intersection(self.BeFixed):
            return SingleActor(act, bn[act])
        else:
            return FrozenSingleActor(act, bn[act], pas)

    def actors(self, pas):
        bn = self.SC.BN
        g = bn.DAG

        for act in self.BeActors:
            yield act, self.__form_actor(bn, g, act, pas)

    def generate(self, nickname, exo):
        """
        Generate a simulation core with a nickname
        :param nickname: nickname of the generated core
        :param exo: dict, input exogenous variables
        :return:
        """
        pc = ParameterCore(nickname, self, dict(), 0)
        if isinstance(exo, ParameterCore):
            pc.Parent = exo
        else:
            for k, v in exo.items():
                if k in self.BeFixed:
                    pc[k] = v

        prior = 0
        for loci in self.FixedChain:
            loci.fill(pc)
            prior += loci.evaluate(pc)
        pc.LogPrior = prior
        pc.Actors = dict(self.actors(pc))

        return pc

    def set_response(self, imp, fixed, actors, hoist, pc):
        prior = 0
        for loci in self.FixedChain:
            if loci.Name in imp:
                pc.Locus[loci.Name] = imp[loci.Name]
            elif loci.Name in fixed:
                pc.Locus[loci.Name] = loci.sample(pc)
            prior += loci.evaluate(pc)

        pc.LogPrior = prior

        bn = self.SC.BN

        for act in actors:
            pc.Actors[act] = FrozenSingleActor(act, bn[act], pc)

        for chd, acts in hoist.items():
            for act in acts:
                pc.ChildrenActors[chd][act] = FrozenSingleActor(act, bn[act], pc)

    def breed(self, nickname, group, pa):
        if group not in self.Children:
            raise KeyError('No matched group')

        ch_sg = self.SC[group]
        chd = ch_sg.generate(nickname, pa)

        if self.SC.Hoist and pa:
            if group not in pa.ChildrenActors:
                pa.ChildrenActors[group] = dict(ch_sg.actors(chd))
        else:
            chd.Actors = dict(ch_sg.actors(chd))

        return chd

    def __repr__(self):
        return '{}({}|{}|{})->{}'.format(self.Name,
                                         self.BeFixed if self.BeFixed else '.',
                                         self.BeRandom if self.BeRandom else '.',
                                         self.BeActors if self.BeActors else '.',
                                         self.Children)

    def to_json(self):
        return {
            'Name': self.Name,
            'Listening': list(self.Listening),
            'BeFixed': list(self.BeFixed),
            'BeRandom': list(self.BeRandom),
            'BeActors': list(self.BeActors),
            'Children': [chd.to_json() for chd in self.Children]
        }
