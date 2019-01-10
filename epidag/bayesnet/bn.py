import epidag as dag
from epidag.bayesnet.loci import *
import re
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

__author__ = 'TimeWz667'
__all__ = ['BayesianNetwork', 'bayes_net_from_json', 'bayes_net_from_script']


def find_order(bn):
    return list(nx.topological_sort(bn.DAG))


def find_roots(bn):
    return [k for k, v in bn.DAG.pred.items() if len(v) is 0]


def find_leaves(bn):
    return [k for k, v in bn.DAG.succ.items() if len(v) is 0]


def find_exo(bn):
    return [k for k, v in bn.DAG.nodes.data() if isinstance(v['loci'], ExoValueLoci)]


def form_js(bn):
    nodes = [bn.DAG.nodes[node]['loci'].to_json() for node in bn.DAG.nodes()]
    return {
        'Name': bn.Name,
        'Nodes': nodes,
        'Order': bn.Order,
        'Roots': bn.Roots,
        'Leaves': bn.Leaves,
        'Exo': bn.Exo
    }


def form_script(bn):
    scr = 'PCore {} '.format(bn.Name) + '{\n'
    for node, dat in bn.DAG.nodes.data():
        scr += '\t' + repr(dat['loci'])
        if 'Des' in dat:
            scr += " # {}".format(dat['Des'])
        scr += '\n'
    scr += '}'
    return scr


class BayesianNetwork:
    def __init__(self, name):
        self.Name = name
        self.DAG = nx.DiGraph()
        self.json = None
        self.script = None
        self.__order = None
        self.__roots = None
        self.__leaves = None
        self.__exo = None

    def append_loci(self, loci, **kwargs):
        if nx.is_frozen(self.DAG):
            return  # todo raise AttributeError

        name = loci.Name
        if name in self.DAG:
            if not isinstance(self.DAG.nodes[name]['loci'], ExoValueLoci):
                raise KeyError('Duplicated variable name')

        self.DAG.add_node(name, loci=loci, **kwargs)

        new_pa = list()
        for pa in loci.Parents:
            if pa not in self.DAG:
                self.append_loci(ExoValueLoci(pa))
                new_pa.append(pa)
            self.DAG.add_edge(pa, name)

        # Check acyclic or not
        if not nx.is_directed_acyclic_graph(self.DAG):
            self.DAG.remove_node(name)
            for pa in new_pa:
                self.DAG.remove_node(pa)
            raise AttributeError('The node causes cyclic paths')

    def append_from_js(self, js):
        loci = loci_from_json(js)
        if 'Des' in js:
            self.append_loci(loci, Des=js['Des'])
        else:
            self.append_loci(loci)

    def append_from_definition(self, df):
        try:
            loci, des = parse_loci(df)
        except dag.ScriptException as e:
            raise e
        if des:
            self.append_loci(loci, Des=des)
        else:
            self.append_loci(loci)

    def complete(self):
        nx.freeze(self.DAG)
        self.__order = find_order(self)
        self.__roots = find_roots(self)
        self.__leaves = find_leaves(self)
        self.__exo = find_exo(self)
        self.json = form_js(self)
        self.script = form_script(self)

    def is_frozen(self):
        return nx.is_frozen(self.DAG)

    @property
    def Order(self):
        return self.__order if self.is_frozen() else find_order(self)

    @property
    def Roots(self):
        return self.__roots if self.is_frozen() else find_roots(self)

    @property
    def Leaves(self):
        return self.__leaves if self.is_frozen() else find_leaves(self)

    @property
    def Exo(self):
        return self.__exo if self.is_frozen() else find_exo(self)

    def needs_calculation(self, node):
        return isinstance(self[node], DistributionLoci) or isinstance(self[node], FunctionLoci)

    def is_rv(self, node):
        return isinstance(self[node], DistributionLoci)

    def __getitem__(self, item):
        return self.DAG.nodes[item]['loci']

    def sort(self, nodes):
        return [node for node in self.Order if node in nodes]

    def clone(self):
        return bayes_net_from_json(self.to_json())

    def to_json(self):
        return self.json if self.is_frozen() else form_js(self)

    def to_script(self):
        return self.script if self.is_frozen() else form_script(self)

    def __str__(self):
        return self.to_script()

    def __repr__(self):
        return 'BayesNet(Name: {}, Nodes: {})'.format(self.Name, self.Order)

    def plot(self):
        pos = graphviz_layout(self.DAG, prog='dot')
        nx.draw(self.DAG, pos, with_labels=True, arrows=True)


def bayes_net_from_script(script):
    """
    Build a Bayesian network from script input
    :param script: multi-line string, script of a Bayesian network
    :return: BayesianNetwork
    """
    lines = script.split('\n')
    lines = [line.replace(' ', '') for line in lines]
    for line in lines:
        mat = re.match(r'pcore(\w+){', line, re.I)
        if mat:
            bn = BayesianNetwork(mat.group(1))
            break
    else:
        raise dag.ScriptException('Unknown script format')
    for line in lines:
        try:
            bn.append_from_definition(line)
        except dag.ScriptException:
            continue
    bn.complete()
    bn.script = script
    return bn


def bayes_net_from_json(js):
    """
    Build a Bayesian network from json input
    :param js: json, json formatted Bayesian network
    :return: BayesianNetwork
    """
    bn = BayesianNetwork(js['Name'])

    for node in js['Nodes']:
        bn.append_from_js(node)
    nx.freeze(bn)
    bn.json = js
    bn.__order = js['Order']
    bn.__roots = js['Roots']
    bn.__leaves = js['Leaves']
    bn.__exo = js['Exo']
    return bn
