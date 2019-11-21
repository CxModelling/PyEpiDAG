from epidag.simulation.parcore import ParameterCore
from epidag.simulation.actor import FrozenSingleActor

__author__ = 'TimeWz667'


class SimulationGroup:
    def __init__(self, ns):
        self.Name = ns.Name
        self.SC = None
        self.BN = None
        self.Children = list(ns.Children.keys())
        self.Listening = list(ns.ListeningNodes)
        self.Exogenous = list(ns.ExoNodes)
        self.Fixed = list(ns.FixedNodes)
        self.Floating = list(ns.FloatingNodes)
        self.LocalActors = ns.LocalSamplers
        self.SharedActors = ns.SharedSamplers

    def set_simulation_core(self, sc):
        self.SC = sc
        self.BN = self.SC.BN

    def generate(self, nickname, parent=None, exo=None):
        """
        Generate a simulation core with a nickname
        :param nickname: nickname of the generated core
        :param parent: ParameterCore, parent Parameter
        :param exo: dict, input exogenous variables
        :return:
        """
        exo = dict(exo) if exo else dict()
        vs = dict(exo)
        for d in self.Exogenous:
            if d not in vs:
                try:
                    vs[d] = parent[d]
                except (TypeError, KeyError):
                    vs[d] = self.BN[d].fill(vs)

        prior = 0
        for d in self.Fixed:
            loci = self.BN[d]
            if d not in vs:
                loci.fill(vs)
            prior += loci.evaluate(vs)

        vs = {d: vs[d] for d in self.Fixed}
        vs.update(exo)

        pc = ParameterCore(nickname, self, vs, prior)
        if parent is not None:
            pc.Parent = parent

        return pc

    def put_local_actors(self, pc):
        actors = {k: v.compose_actor(self.BN) for k, v in self.LocalActors.items()}
        for k, v in actors.items():
            if isinstance(v, FrozenSingleActor):
                v.update(pc)
        pc.Actors = actors

    def put_shared_actors_on_parent(self, parent):
        actors = {k: v.compose_actor(self.BN) for k, v in self.SharedActors.items()}
        for k, v in actors.items():
            if isinstance(v, FrozenSingleActor):
                v.update(parent)
        if parent.ChildrenActors is None:
            parent.ChildrenActors = dict()
        parent.ChildrenActors[self.Name] = actors

    def put_shared_actors(self, pc):
        parent = pc.Parent
        if parent is None:
            raise AttributeError('Root node cannot share samplers')
        self.put_shared_actors_on_parent(parent)

    def set_response(self, imp, shocked, actors, hoist, pc):
        prior = 0
        for d in self.Fixed:
            loci = self.BN[d]
            if d in imp:
                pc.Locus[d] = imp[d]
            elif d in shocked:
                loci.fill(pc)
            prior += loci.evaluate(pc)

        pc.LogPrior = prior

        for act in actors:
            pc.Actors[act].update(pc)

        for k, vs in hoist.items():
            for act in vs:
                pc.ChildrenActors[k][act].update(pc)

    def breed(self, nickname, group, pa, exo):
        if group not in self.Children:
            raise KeyError('No matched group')

        chd = self.SC[group].generate(nickname, parent=pa, exo=exo)

        return chd

    def __repr__(self):
        return '{}({}|{}|{})->{}'.format(self.Name,
                                         self.Exogenous if self.Exogenous else '.',
                                         self.Listening if self.Listening else '.',
                                         self.Fixed if self.Fixed else '.',
                                         self.Floating if self.Floating else '.',
                                         self.Children)

    def to_json(self):
        return {
            'Name': self.Name,
            'Exogenous': list(self.Exogenous),
            'Listening': list(self.Listening),
            'BeFixed': list(self.Fixed),
            'BeFloating': list(self.Floating),
            'Children': list(self.Children)
        }
