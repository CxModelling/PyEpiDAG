import networkx as nx

__author__ = 'TimeWz667'
__all__ = ['get_sufficient_nodes']


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


def form_hierarchy(bn, hie=None, root='root'):
    """

    :param bn: epidag.BayesNet, a Bayesian Network
    :param hie: hierarical structure of the nodes of bn
    :param root: name of root group
    :return: A tree structure with NodeGroup
    """
    g = bn.DAG

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

    # check order
    if isinstance(hie, dict):
        root = define_node(root, hie)

    elif isinstance(hie, list):
        root = NodeGroup(root, hie[0])
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
