from epidag.simulation.simugroup import SimulationGroup
from epidag.simulation.nodeset import NodeSet

__author__ = 'TimeWz667'

__all__ = ['SimulationCore']


def get_simulation_groups(root: NodeSet):
    sgs = dict()

    def set_gp(ns: NodeSet):
        sgs[ns.Name] = SimulationGroup(ns)
        if ns.Children:
            for chd in ns.Children.values():
                set_gp(chd)

    set_gp(root)

    return sgs


class SimulationCore:
    def __init__(self, bn, root=None):
        self.Name = bn.Name
        self.BN = bn
        self.RootNode = root
        self.RootSG = root.Name
        self.SGs = get_simulation_groups(root)
        for sg in self.SGs.values():
            sg.set_simulation_core(self)

    def __getitem__(self, item):
        return self.SGs[item]

    def get(self, item):
        try:
            return self.SGs[item]
        except KeyError:
            raise KeyError('Unknown group')

    def generate(self, nickname=None, exo=None):
        """
        Instantiate a simulation model
        :param nickname: nickname of generated parameter
        :param exo: dict, exogenous variables
        :return:
        """
        nickname = nickname if nickname else self.Name
        exo = dict(exo) if exo else dict()
        return self.SGs[self.RootSG].generate(nickname, None, exo)

    def to_json(self):
        return {
            'BayesianNetwork': self.BN.to_json(),
            'Root': self.RootSG
        }

    def deep_print(self):
        self.RootNode.print()

    def clone(self):
        return SimulationCore(self.BN, self.RootNode)

    def __repr__(self):
        return 'Simulation core: {}'.format(self.Name)
