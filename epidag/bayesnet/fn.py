import networkx as nx
from .loci import ValueLoci, ExoValueLoci

__author__ = 'TimeWz667'
__all__ = ['get_sufficient_nodes', 'form_hierarchy', 'formulate_blueprint', 'analyse_node_type']


def get_sufficient_nodes(g, included, given):
    """
    Find the required nodes to assess the included nodes
    :param g: a directed acyclic graph
    :param included: targeted nodes
    :param given: certain nodes
    :return: a set of the required nodes
    """
    mi = g.copy()
    # remove all parents from the given nodes
    for nod in given:
        pas = list(mi.predecessors(nod))
        for pa in pas:
            mi.remove_edge(pa, nod)
    # find the nodes supporting the included nodes
    return set.union(*[nx.ancestors(mi, nod) for nod in included]).union(included)


class NodeGroup:
    def __init__(self, name, fixed):
        self.Name = name
        self.Children = set()
        self.Nodes = set(fixed)

    def append_chd(self, chd):
        self.Children.add(chd)

    def needs(self, nod, g):
        if any(nod in nx.ancestors(g, x) for x in self.Nodes):
            return True
        elif any(nod in nx.descendants(g, x) for x in self.Nodes):
            return True
        elif any(chd.needs(nod, g) for chd in self.Children):
            return True
        else:
            return False

    def can_be_passed_down(self, nod, g):
        des = nx.descendants(g, nod)
        if any(x in des for x in self.Nodes):
            return False
        else:
            return True

    def catch(self, nod):
        self.Nodes.add(nod)

    def pop(self, nod):
        self.Nodes.remove(nod)

    def pass_down(self, nod, g):
        self.catch(nod)
        if not self.can_be_passed_down(nod, g):
            return

        needed = [chd for chd in self.Children if chd.needs(nod, g)]
        if len(needed) is 1:
            self.pop(nod)
            needed[0].pass_down(nod, g)

    def has(self, nod):
        if nod in self.Nodes:
            return True
        if any(chd.has(nod) for chd in self.Children):
            return True
        else:
            return False

    def can_be_raised_up(self, nod, g):
        anc = nx.ancestors(g, nod)
        if any(x in anc for x in self.Nodes):
            return False
        else:
            return True

    def raise_up(self, nod, g):
        for chd in self.Children:
            if not chd.has(nod):
                continue
            if chd.can_be_raised_up(nod, g):
                chd.raise_up(nod, g)
                chd.pop(nod)
                self.catch(nod)
                return

    def get_all(self):
        return set.union(self.Nodes, *[chd.get_all() for chd in self.Children])

    def print(self, i=0):
        print('{}{}{}({})'.format('--'*i, ' ' if i else '', self.Name, ', '.join(self.Nodes)))
        for chd in self.Children:
            chd.print(i + 1)


def divide(xs, key):
    tr, fa = list(), list()
    for x in xs:
        if key(x):
            tr.append(x)
        else:
            fa.append(x)
    return tr, fa


def define_node(k, links):
    chd, node = divide(links[k], lambda x: x in links)
    curr = NodeGroup(k, node)
    for ch in chd:
        curr.append_chd(define_node(ch, links))
    return curr


def form_hierarchy(bn, hie=None, root=None):
    """

    :param bn: epidag.BayesNet, a Bayesian Network
    :param hie: hierarchical structure of the nodes of bn
    :param root: name of root group
    :return: A tree structure with NodeGroup
    """
    g = bn.DAG

    # find the root group of node networks
    if isinstance(hie, dict):
        root_p = list()

        for k in hie.keys():
            if not (any([k in v for v in hie.values()])):
                root_p.append(k)

        if len(root_p) is 1:
            root = define_node(root_p[0], hie)
        elif len(root_p) is 0:
            raise SyntaxError('Hierarchy is ill-defined')
        else:
            if 'root' in hie:
                hie['root'] += [k for k in root_p if k != 'root']
            else:
                hie['root'] = root_p
            root = define_node('root', hie)

    elif isinstance(hie, list):
        root = NodeGroup('root', hie[0])
        ng1 = root
        for i, hi in enumerate(hie[1:], 2):
            ng0, ng1 = ng1, NodeGroup('Layer {}'.format(i), hi)
            ng0.append_chd(ng1)
    else:
        root = NodeGroup(root, bn.OrderedNodes)

    # root.print()

    all_fixed = root.get_all()
    all_floated = [nod for nod in bn.OrderedNodes if nod not in all_fixed]

    all_floated.reverse()

    for nod in bn.ExogenousNodes:
        root.catch(nod)

    all_floated = [nod for nod in all_floated if nod not in bn.ExogenousNodes]
    for nod in all_floated:
        root.pass_down(nod, g)

    all_floated.reverse()
    for nod in all_floated:
        root.raise_up(nod, g)

    return root


def analyse_node_type(bn, root=None, report=False):
    """
    Analyse nodes in each group based on their potential characteristics.
    A node which can be an actor must be a leaf of the given DAG.
    A node which can carry stochastic effects must not be an ancestor of nodes in lower models.

    :param bn: epidag.BayesNet, a Bayesian Network
    :param root: root node group
    :param report: True if report print needed
    :return: dict, key = Group name, value = (Should be fixed, Can be random, Can be actors)
    """
    g = bn.DAG
    leaves = bn.LeafNodes

    if not root:
        root = form_hierarchy(bn)

    def no_randomness(nod):
        return isinstance(nod, ValueLoci) or isinstance(nod, ExoValueLoci)

    res = dict()
    must_fixed = [k for k, v in g.nodes().items() if no_randomness(v['loci'])]

    def fn(ng, ind=0):
        fix, ran, act = list(), list(), list()
        for node in ng.Nodes:
            if node in must_fixed:
                fix.append(node)
            elif node in leaves:
                act.append(node)
            elif nx.descendants(g, node) <= ng.Nodes:
                ran.append(node)
            else:
                fix.append(node)

        res[ng.Name] = fix, ran, act
        if report:
            print('{}Group {}'.format('--' * ind, ng.Name))
            print('{}Must be fixed: {}'.format('  ' * ind, fix))
            print('{}Can be random: {}'.format('  ' * ind, ran))
            print('{}Can be actors: {}'.format('  ' * ind, act))

        for chd in ng.Children:
            fn(chd, ind + 1)

    fn(root)
    return res


def formulate_blueprint(bn, root=None, random=None, out=None):
    """
    a blueprint of a simulation model based on given a Bayesian network.
    It describes every node in the network as 1) fixed variable, 2) random variable, 3) exposed distribution
    :param bn: epidag.BayesNet, a Bayesian Network
    :param root: root node group
    :param random: nodes with random effects within an individual
    :param out: nodes can be used in simulation model
    :return: a blueprint of simulation model
    """
    suggest = analyse_node_type(bn, root, report=False)
    random = random if random else list()
    out = out if out else set.union(*[set(act) for (_, _, act) in suggest.values()])
    out = [o for o in out if o not in random]

    approved = dict()
    for k, (fs, rs, cs) in suggest.items():
        afs = list(fs)

        ars = list()
        for r in rs:
            if r in random:
                ars.append(r)
            else:
                afs.append(r)

        acs = list()
        for c in cs:
            if c in out:
                acs.append(c)
            elif c in random:
                ars.append(c)
            else:
                afs.append(c)

        approved[k] = afs, ars, acs
    # todo detect unwilling changes
    return approved
