from .simugroup import SimulationGroup
__author__ = 'TimeWz667'

__all__ = ['SimulationCore']


def get_simulation_groups(bn, bp, root):
    g = bn.DAG

    sgs = dict()
    for k, (fs, rs, cs) in bp.items():
        nodes = set(fs + rs + cs)
        pas = set.union(*[set(g.predecessors(node)) for node in nodes])
        pas = pas - nodes
        sgs[k] = SimulationGroup(k, fs, rs, cs, pas)

    def set_children(ng):
        sg = sgs[ng.Name]
        for chd in ng.Children:
            sg.Children.append(chd.Name)
            set_children(chd)

    set_children(root)

    return sgs


class SimulationCore:
    def __init__(self, bn, bp=None, root=None, hoist=True):
        self.Name = bn.Name
        self.BN = bn
        self.RootSG = root.Name
        self.SGs = get_simulation_groups(bn, bp, root)
        for sg in self.SGs.values():
            sg.set_simulation_core(self)
        self.Hoist = hoist

    def __getitem__(self, item):
        return self.SGs[item]

    def get(self, item):
        try:
            return self.SGs[item]
        except KeyError:
            raise KeyError('Unknown group')

    def generate(self, nickname, exo=None):
        """
        Instantiate a simulation model
        :param nickname: nickname of generated parameter
        :param exo: dict, exogenous variables
        :return:
        """
        return self.SGs[self.RootSG].generate(nickname, exo)

    def to_json(self):
        return {
            'BayesianNetwork': self.BN.to_json(),
            'Root': self.RootSG
        }

    def __repr__(self):
        return 'Simulation core: {}'.format(self.Name)

