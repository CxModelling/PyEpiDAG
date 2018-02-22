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
