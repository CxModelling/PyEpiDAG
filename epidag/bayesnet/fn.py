import networkx as nx
from epidag.bayesnet.loci import ValueLoci, ExoValueLoci
import numpy as np

__author__ = 'TimeWz667'
__all__ = ['NodeGroup',
           'get_sufficient_nodes', 'get_minimal_nodes',
           'form_hierarchy', 'formulate_blueprint',
           'analyse_node_type', 'evaluate_nodes']


def get_sufficient_nodes(g, included, given=None):
    """
    Find the required nodes to assess the included nodes
    :param g: a directed acyclic graph
    :param included: targeted nodes
    :param given: certain nodes
    :return: a set of the required nodes
    """
    mi = g.copy()
    given = given if given else dict()
    # remove all parents from the given nodes
    for nod in given:
        pas = list(mi.predecessors(nod))
        for pa in pas:
            mi.remove_edge(pa, nod)
    # find the nodes supporting the included nodes
    return set.union(*[nx.ancestors(mi, nod) for nod in included]).union(included)


def get_minimal_nodes(g, included, given=None):
    """
    Find the required nodes to assess the included nodes without considering given nodes
    :param g: a directed acyclic graph
    :param included: targeted nodes
    :param given: certain nodes
    :return: a set of the required nodes
    """
    suf = get_sufficient_nodes(g, included=included, given=given)
    suf.difference_update(given)
    return suf


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

    def deep_remove(self, k):
        try:
            self.Nodes.remove(k)
        except ValueError:
            pass
        for chd in self.Children:
            chd.deep_remove(k)

    def check_conflict(self, g):
        for chd in self.Children:
            chd.check_conflict(g)

        for c1 in self.Children:
            nodes = c1.get_all()
            for c2 in self.Children:
                if c1 is c2:
                    continue
                for node in nodes:
                    if c2.needs(node, g):
                        c1.deep_remove(node)
                        self.Nodes.add(node)

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

    :param bn: a Bayesian Network
    :type: BayesNet
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
        root = NodeGroup(root, bn.Order)

    # root.print()
    root.check_conflict(g)
    all_fixed = root.get_all()
    all_floated = [nod for nod in bn.Order if nod not in all_fixed]

    all_floated.reverse()

    # for nod in bn.ExogenousNodes:
    #    root.catch(nod)

    all_floated = [nod for nod in all_floated]  # if nod not in bn.ExogenousNodes]
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

    :param bn: BayesNet, a Bayesian Network
    :param bn: BayesNet
    :param root: root node group
    :param report: True if report print needed
    :type report: bool
    :return: key = Group name, value = (Should be fixed, Can be random, Can be actors)
    :rtype: dict
    """
    g = bn.DAG

    if not root:
        root = form_hierarchy(bn)

    res = dict()
    must_fixed = [k for k, v in g.nodes().items() if isinstance(v['loci'], ValueLoci)]
    exogenous = [k for k, v in g.nodes().items() if isinstance(v['loci'], ExoValueLoci)]

    def fn(ng, ind=0):
        exo, fix, fr, fra, ra, rd = list(), list(), list(), list(), list(), list()
        for node in ng.Nodes:
            if node in exogenous:
                exo.append(node)
            elif node in must_fixed:
                fix.append(node)
            else:
                des = nx.descendants(g, node)
                ans = nx.ancestors(g, node)
                if des:
                    if des <= ng.Nodes:
                        if set.intersection(ans, exogenous):
                            rd.append(node)
                        else:
                            fr.append(node)
                    else:
                        fix.append(node)
                else:
                    if set.intersection(ans, exogenous):
                        ra.append(node)
                    else:
                        fra.append(node)

        res[ng.Name] = exo, fix, fr, fra, ra, rd
        if report:
            print('{}Group {}'.format('--' * ind, ng.Name))
            print('{}Exogenous    : {}'.format('  ' * ind, exo))
            print('{}Can be fixed: {}'.format('  ' * ind, fix + fr + fra))
            print('{}Can be random: {}'.format('  ' * ind, fr + fra + ra + rd))
            print('{}Can be actors: {}'.format('  ' * ind, fra + ra))

        for chd in ng.Children:
            fn(chd, ind + 1)

    fn(root)
    return res


def formulate_blueprint(bn, root=None, random=None, out=None):
    """
    a blueprint of a simulation model based on given a Bayesian network.
    It describes every node in the network as 1) fixed variable, 2) random variable, 3) exposed distribution
    :param bn: a Bayesian Network
    :type bn: BayesNet
    :param root: root node group
    :param random: nodes with random effects within an individual
    :param out: nodes can be used in simulation model
    :return: a blueprint of simulation model
    """
    suggest = analyse_node_type(bn, root, report=False)
    random = random if random else list()
    if out is None:
        out = set.union(*[set(fra + ra) for (_, _, _, fra, ra, _) in suggest.values()])
        out = set([o for o in out if bn.needs_calculation(o)])
    elif len(out) is 0:
        out = []

    out = [o for o in out if o not in random]

    approved = dict()
    for k, (exo, fix, fr, fra, ra, rd) in suggest.items():
        aes = list(exo)
        afs = list(fix)
        ars = list(rd)
        acs = list()
        for r in fr:
            if r in random:
                ars.append(r)
            else:
                afs.append(r)

        for r in fra:
            if r in out:
                acs.append(r)
            elif r in random:
                ars.append(r)
            else:
                afs.append(r)

        for r in ra:
            if r in out:
                acs.append(r)
            elif r in random:
                ars.append(r)
            else:
                raise(TypeError('{} can not be fixed'.format(r)))

        approved[k] = aes, afs, ars, acs
    return approved


def evaluate_nodes(bn, pars):
    """
    Evaluate the likelihood of a set of variables
    :param bn: epidag.BayesNet, a Bayesian Network
    :param pars: dict, a container of parameters
    :return: the log likelihood of pars
    """
    nodes = bn.DAG.nodes
    lps = np.sum([nodes[k]['loci'].evaluate(pars) for k in pars.keys()])
    return lps
