import networkx as nx


__author__ = 'TimeWz'
__all__ = ['DAG', 'merge_dag', 'minimal_dag', 'minimal_requirements']



class DAG(nx.DiGraph):
    def __init__(self, data=None, **attr):
        nx.DiGraph.__init__(self, incoming_graph_data=data, **attr)

    def parents(self, node):
        return set(self.predecessors(node))

    def children(self, node):
        return set(self.successors(node))

    def ancestors(self, node):
        return set(nx.ancestors(self, node))

    def descendants(self, node):
        return set(nx.descendants(self, node))

    def roots(self):
        return [k for k, v in self.pred.items() if len(v) is 0]

    def leaves(self):
        return [k for k, v in self.succ.items() if len(v) is 0]

    def order(self):
        return list(nx.topological_sort(self))

    def sort(self, nodes):
        return [node for node in self.order() if node in nodes]

    def check_acyclic(self):
        return nx.is_directed_acyclic_graph(self)

    def remove_out_edges(self, node):
        for chd in self.children(node):
            self.remove_edge(node, chd)

    def remove_in_edges(self, node):
        for par in self.parents(node):
            self.remove_edge(node, par)

    def remove_upstream(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        nodes = set(nodes)
        ord = self.order()
        for s in ord:
            if s in nodes:
                continue
            if nodes.intersection(self.ancestors(s)):
                self.remove_node(s)


    def remove_downstream(self, nodes):
        if not isinstance(nodes, list):
            nodes = [nodes]
        nodes = set(nodes)
        ord = self.order()
        ord.reverse()
        for s in ord:
            if s in nodes:
                continue
            if nodes.intersection(self.descendants(s)):
                self.remove_node(s)


def merge_dag(dag1, dag2):
    return nx.compose(dag1, dag2)


def minimal_dag(dag, nodes):
    nodes = dag.sort(nodes)
    to_collect = set(nodes
                     )
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            for pth in nx.all_simple_paths(dag, nodes[i], nodes[j]):
                to_collect.update(pth)

    return dag.subgraph(to_collect)


def minimal_requirements(dag, target, cond):
    anc = set(dag.ancestors(target))

    for s in cond:
        if s in anc:
            anc.difference_update(dag.ancestors(s))

    return dag.sort(anc)



if __name__ == '__main__':
    a = DAG()
    a.add_edge('A', 'B')
    a.add_edge('B', 'C')
    print(a.descendants('A'))
    a.nodes['A']['loci'] = [1,2,3]


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

    print(type(minimal_dag(c, ['A', 'C'])))

    print(minimal_requirements(c, 'D', ['C']))
