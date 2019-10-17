import epidag as dag
from epidag.bayesnet.loci import *
from epidag.bayesnet.dag import DAG, merge_dag
import re
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

__author__ = 'TimeWz667'
__all__ = ['BayesianNetwork', 'bayes_net_from_json', 'bayes_net_from_script']


def find_rv_roots(bn):
    rr = list()

    for k in bn.Order:
        if bn.is_rv(k):
            for a in nx.ancestors(bn.DAG, k):
                if bn.is_rv(a):
                    break
            else:
                rr.append(k)
    return rr


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
        self.DAG = DAG()
        self.UserDefinedFunctions = dict()
        self.json = None
        self.script = None
        self.__order = None
        self.__roots = None
        self.__rv_roots = None
        self.__leaves = None
        self.__exo = None

    def append_loci(self, loci, **kwargs):
        if nx.is_frozen(self.DAG):
            raise AttributeError('The structure has been fixed')

        name = loci.Name
        if name in self.DAG:
            if not isinstance(self.DAG.nodes[name]['loci'], ExoValueLoci):
                raise KeyError('Duplicated variable name')

        self.DAG.add_node(name, loci=loci, **kwargs)
        self.DAG.remove_in_edges(name)

        new_pa = list()
        for pa in loci.Parents:
            if pa not in self.DAG:
                self.append_loci(ExoValueLoci(pa))
                new_pa.append(pa)
            self.DAG.add_edge(pa, name)

        # Check acyclic or not
        if not self.DAG.check_acyclic():
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

    def add_user_defined_func(self, fn_name, fn):
        assert fn_name not in self.UserDefinedFunctions
        self.UserDefinedFunctions[fn_name] = fn

    def __check_edges(self):
        exo = set()
        for node in self.DAG.order():
            loci = self[node]
            pars = set(loci.Parents)
            for k in list(self.DAG.parents(node)):
                if k not in pars:
                    self.DAG.remove_edge(k, node)
            pars.difference_update(self.DAG.parents(node))
            exo.update(pars)

        for node in exo:
            self.append_loci(ExoValueLoci(node))

        exo = find_exo(self)
        for node in exo:
            if not self.DAG.children(node):
                self.DAG.remove_node(node)

    def complete(self):
        nx.freeze(self.DAG)
        self.__order = self.DAG.order()
        self.__roots = self.DAG.roots()
        self.__rv_roots = find_rv_roots(self)
        self.__leaves = self.DAG.leaves()
        self.__exo = find_exo(self)
        self.json = form_js(self)
        self.script = form_script(self)

    def defrost(self):
        self.DAG = DAG(self.DAG)
        self.json = None
        self.script = None
        self.__order = None
        self.__roots = None
        self.__rv_roots = None
        self.__leaves = None
        self.__exo = None

    def is_frozen(self):
        return nx.is_frozen(self.DAG)

    @property
    def Order(self):
        return self.__order if self.is_frozen() else self.DAG.order()

    @property
    def Roots(self):
        return self.__roots if self.is_frozen() else self.DAG.roots()

    @property
    def RVRoots(self):
        return self.__rv_roots if self.is_frozen() else find_rv_roots(self)

    @property
    def Leaves(self):
        return self.__leaves if self.is_frozen() else self.DAG.leaves()

    @property
    def Exo(self):
        return self.__exo if self.is_frozen() else find_exo(self)

    def needs_calculation(self, node):
        return isinstance(self[node], DistributionLoci) or isinstance(self[node], FunctionLoci)

    def is_exogenous(self, node):
        node = self[node]
        return isinstance(node, PseudoLoci) or isinstance(node, ExoValueLoci)

    def is_rv(self, node):
        return isinstance(self[node], DistributionLoci)

    def is_deterministic(self, node, given=None):
        if isinstance(self[node], ValueLoci):
            return True
        if given:
            req = dag.minimal_requirements(self.DAG, node, given)
            req = [d for d in req if d not in given]
            return all(isinstance(self[d], ValueLoci) for d in req)
        else:
            return False

    def has_randomness(self, node, given=None):
        if self.is_rv(node):
            return True
        if given:
            req = dag.minimal_requirements(self.DAG, node, given)
            req = [d for d in req if d not in given]
        else:
            req = self.DAG.ancestors(node)

        for d in req:
            if self.is_rv(d):
                return True
        else:
            return False


    def __getitem__(self, item):
        return self.DAG.nodes[item]['loci']

    def __contains__(self, item):
        return item in self.DAG

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

    def merge(self, name, sub_bn):
        assert  name != self.Name and name != sub_bn.Name

        bn = self.copy(name)
        bn.defrost()
        for node in sub_bn.Order:
            if sub_bn.is_exogenous(node):
                continue
            if node in bn:
                bn.DAG.remove_node(node)
            bn.append_from_js(sub_bn[node].to_json())

        bn.__check_edges()
        bn.UserDefinedFunctions.update(self.UserDefinedFunctions)
        bn.UserDefinedFunctions.update(sub_bn.UserDefinedFunctions)
        return bn

    def copy(self, new_name=None):
        if not new_name:
            new_name = self.Name
        bn = BayesianNetwork(new_name)

        if self.is_frozen():
            for node in self.json['Nodes']:
                bn.append_from_js(node)

            bn.complete()

        else:
            for node in self.DAG.nodes():
                bn.append_from_js(self[node].to_json())
        bn.UserDefinedFunctions.update(self.UserDefinedFunctions)
        return bn

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
    bn.__rv_roots = js['RVRoots']
    bn.__leaves = js['Leaves']
    bn.__exo = js['Exo']
    return bn
