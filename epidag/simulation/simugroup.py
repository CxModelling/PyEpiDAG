from epidag.simulation.parcore import ParameterCore

__author__ = 'TimeWz667'


class SimulationGroup:
    def __init__(self, ns, hoist=True):
        self.Name = ns.Name
        self.SC = None
        self.BN = None
        self.Children = list(ns.Children.keys())
        self.Hoisting = hoist
        self.Listening = list(ns.ListeningNodes)
        self.Exogenous = list(ns.ExoNodes)
        self.Fixed = list(ns.FixedNodes)
        self.Floating = list(ns.FloatingNodes)
        self.Actors = ns.Samplers
        self.ChildrenActors = dict(ns.ChildrenSamplers)

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
                vs[d] = parent[d]

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

        self.set_actors(pc)
        return pc

    def set_actors(self, pc):
        if not pc.Parent or not self.Hoisting:
            pc.Actors = dict()
            for k, act in self.Actors.items():
                pc.Actors[k] = act.compose_actor(self.BN)
        else:
            self.set_child_actors(pc.Parent, pc.Group)

    def set_child_actors(self, parent, sg):
        if parent.ChildrenActors is None:
            parent.ChildrenActors = dict()

        if sg in parent.ChildrenActors:
            return

        actors = dict()
        bps = self.SC.get(parent.Group).ChildrenActors[sg]
        for k, act in bps.items():
            try:
                actors[k] = act.compose_actor(self.BN)
            except AttributeError:
                continue
        parent.ChildrenActors[sg] = actors

    def update_actors(self, pc, pars):
        pass

    def set_response(self, imp, shocked, actors, hoist, pc):
        prior = 0
        for d in self.Fixed:
            loci = self.BN[d]
            if d in shocked:
                try:
                    pc.Locus[d] = imp[d]
                except KeyError:
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
