import networkx as nx

__author__ = 'TimeWz'
__all__ = ['DAG', 'merge_dag', 'minimal_dag', 'minimal_requirements',
           'get_sufficient_nodes', 'get_minimal_nodes', 'get_offsprings']


class DAG(nx.DiGraph):
    def __init__(self, data=None, **attr):
        nx.DiGraph.__init__(self, incoming_graph_data=data, **attr)

    def parents(self, node):
        """
        Find the parent nodes
        :param node: target node
        :return: a set of parent nodes
        :rtype: set
        """
        return set(self.predecessors(node))

    def children(self, node):
        """
        Find the child nodes
        :param node: target node
        :return: a set of child nodes
        :rtype: set
        """
        return set(self.successors(node))

    def ancestors(self, node):
        """
        Find the ancestor nodes
        :param node: target node
        :return: a set of ancestor nodes
        :rtype: set
        """
        return set(nx.ancestors(self, node))

    def descendants(self, node):
        """
        Find the descendant nodes
        :param node: target node
        :return: a set of descendant nodes
        :rtype: set
        """
        return set(nx.descendants(self, node))

    def upstream(self, nodes):
        """
        Find the offspring nodes in a DAG
        :param nodes: targeted nodes
        :return: a set of offspring nodes
        """
        return set.union(*[self.ancestors(d) for d in nodes])

    def downstream(self, nodes):
        """
        Find the offspring nodes in a DAG
        :param nodes: targeted nodes
        :return: a set of offspring nodes
        """
        return set.union(*[self.descendants(d) for d in nodes])

    def roots(self):
        """
        Find the nodes without parents
        :return: a list of root nodes
        """
        return [k for k, v in self.pred.items() if len(v) is 0]

    def leaves(self):
        """
        Find the nodes without children
        :return: a list of leaf nodes
        """
        return [k for k, v in self.succ.items() if len(v) is 0]

    def order(self):
        """
        Find the order of nodes
        :return: a list of sorted nodes
        """
        return list(nx.topological_sort(self))

    def sort(self, nodes):
        """
        Find the order of nodes
        :return: a list of sorted nodes
        """
        return [node for node in self.order() if node in nodes]

    def check_acyclic(self):
        """
        Check if the DAG is acyclic or not
        :return: true if validated
        """
        return nx.is_directed_acyclic_graph(self)

    def remove_out_edges(self, node):
        """
        Remove impacts on children
        :param node: targeted node
        """
        for chd in self.children(node):
            self.remove_edge(node, chd)

    def remove_in_edges(self, node):
        """
        Remove incoming edges
        :param node: targeted node
        """
        for par in self.parents(node):
            self.remove_edge(node, par)

    def remove_upstream(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        nodes = set(nodes)
        od = self.order()
        for s in od:
            if s in nodes:
                continue
            if nodes.intersection(self.ancestors(s)):
                self.remove_node(s)

    def remove_downstream(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        nodes = set(nodes)
        od = self.order()
        od.reverse()
        for s in od:
            if s in nodes:
                continue
            if nodes.intersection(self.descendants(s)):
                self.remove_node(s)

    def __str__(self):
        return 'DAG({})'.format(', '.join(self.order()))

    def __repr__(self):
        return 'DAG({})'.format(', '.join(self.order()))


def merge_dag(g1, g2):
    return nx.compose(g1, g2)


def minimal_dag(g, nodes):
    nodes = g.sort(nodes)
    to_collect = set(nodes)

    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            for pth in nx.all_simple_paths(g, nodes[i], nodes[j]):
                to_collect.update(pth)

    return g.subgraph(to_collect)


def minimal_requirements(g, target, cond):
    anc = set(g.ancestors(target))

    for s in cond:
        if s in anc:
            anc.difference_update(g.ancestors(s))

    return g.sort(anc)


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
    Find the required nodes to assess the included nodes without given nodes
    :param g: a directed acyclic graph
    :param included: targeted nodes
    :param given: certain nodes
    :return: a set of the required nodes
    """
    suf = get_sufficient_nodes(g, included=included, given=given)
    suf.difference_update(given)
    return suf


def get_offsprings(g, nodes):
    """
    Find the offspring nodes in a DAG
    :param g: a directed acyclic graph
    :param nodes: targeted nodes
    :return: a set of offspring nodes
    """
    return set.union(*[g.descendants(d) for d in nodes])


if __name__ == '__main__':
    a = DAG()
    a.add_edge('A', 'B')
    a.add_edge('B', 'C')
    print(a.descendants('A'))
    a.nodes['A']['loci'] = [1, 2, 3]

    b = a.copy()
    b.nodes['A']['loci'][2] = 5

    print(b.nodes['A']['loci'])

    print(b.in_degree)

    c = DAG()
    c.add_edge('A', 'B')
    c.add_edge('A', 'C')
    c.add_edge('B', 'C')
    c.add_edge('B', 'D')
    c.add_edge('C', 'D')
    print(c.edges)

    print(c)
    print(type(minimal_dag(c, ['A', 'C'])))

    print(minimal_requirements(c, 'D', ['C']))
