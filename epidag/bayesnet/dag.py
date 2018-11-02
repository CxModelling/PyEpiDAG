import epidag as dag
from epidag.bayesnet.loci import *
import re
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

__author__ = 'TimeWz667'
__all__ = ['bn_script_to_json', 'bn_from_json', 'bn_from_script', 'BayesianNetwork']


def bn_script_to_json(script):
    # remove space
    pars = script.replace(' ', '')
    pars = pars.replace('\t', '')

    # split lines
    pars = pars.split('\n')
    pars = [par.split('#')[0] for par in pars if par != '']

    try:
        name = re.match(r'PCore\s*(?P<name>\w+)\s*\{', pars[0], re.IGNORECASE).group('name')
    except AttributeError:
        raise SyntaxError('Name does not identified')

    nodes = dict()

    all_fu = set()
    all_pa = set()
    for p in pars:
        if p.find('~') >= 0:
            p = re.match(r'(\w+)\~(\S+\((\S+)\))', p, re.IGNORECASE)
            p_name, p_func = p.group(1), p.group(3)
            pas, fu = dag.parse_parents(p_func)
            all_fu = all_fu.union(fu)
            all_pa = all_pa.union(pas)
            nodes[p_name] = {'Name': p_name,
                             'Type': 'Distribution',
                             'Def': p.group(2), 'Parents': list(pas)}
        elif p.find('=') >= 0:
            p = re.match(r'(\w+)\=(\S+)', p, re.IGNORECASE)
            p_name, p_func = p.group(1), p.group(2)
            pseudo = p_func.startswith('f(')
            pas, fu = dag.parse_parents(p_func)
            if not pseudo:
                all_fu = all_fu.union(fu)
            all_pa = all_pa.union(pas)
            if not pas:
                node = {'Type': 'Value', 'Def': p_func}
            elif pseudo:
                node = {'Type': 'Pseudo', 'Def': p_func, 'Parents': list(pas)}
            else:
                node = {'Type': 'Function', 'Def': p_func, 'Parents': list(pas)}
            node['Name'] = p_name
            nodes[p_name] = node

    for pa in all_pa:
        if pa not in nodes:
            nodes[pa] = {'Type': 'ExoValue'}

    return {'Name': name, 'Nodes': nodes, 'Dependency': list(all_fu)}


class BayesianNetwork:
    def __init__(self, js):
        self.Name = js['Name']
        self.Source = js
        self.DAG = nx.DiGraph()

        for k, v in js['Nodes'].items():
            if v['Type'] is 'Value':
                lo = ValueLoci(k, v['Def'])
            elif v['Type'] is 'ExoValue':
                lo = ExoValueLoci(k)
            elif v['Type'] is 'Distribution':
                lo = DistributionLoci(k, v['Def'])
            elif v['Type'] is 'Pseudo':
                lo = PseudoLoci(k, v['Def'])
            else:
                lo = FunctionLoci(k, v['Def'])

            self.DAG.add_node(k, loci=lo, **v)
            if 'Parents' not in v:
                continue
            for pa in v['Parents']:
                self.DAG.add_edge(pa, k)

        if not nx.is_directed_acyclic_graph(self.DAG):
            raise SyntaxError('Cyclic groups found')
        if set(js['Dependency']) > dag.MATH_FUNC.keys():
            raise SyntaxError('Unknown functions found')

        nx.freeze(self.DAG)
        self.ExogenousNodes = [k for k, v in self.DAG.nodes.data() if v['Type'] is 'ExoValue']
        self.RootNodes = [k for k, v in self.DAG.pred.items() if len(v) is 0]
        self.LeafNodes = [k for k, v in self.DAG.succ.items() if len(v) is 0]
        self.OrderedNodes = list(nx.topological_sort(self.DAG))

    def is_rv(self, node):
        return isinstance(self[node], DistributionLoci)

    def __getitem__(self, item):
        return self.DAG.nodes[item]['loci']

    def sort(self, nodes):
        return [node for node in self.OrderedNodes if node in nodes]

    def copy(self):
        return BayesianNetwork(self.Source)

    def to_json(self):
        return self.Source

    def __str__(self):
        ss = ['Name:\t{}'.format(self.Source['Name']), 'Nodes:']
        for k in self.OrderedNodes:
            v = self.DAG.nodes[k]
            ss.append('\t{}'.format(v['loci']))
        return '\n'.join(ss)

    def plot(self):
        pos = graphviz_layout(self.DAG, prog='dot')
        nx.draw(self.DAG, pos, with_labels=True, arrows=True)

    def clone(self):
        return BayesianNetwork(self.to_json())

    __repr__ = __str__


def bn_from_script(script):
    """
    Build a Bayesian network from script input
    :param script: multi-line string, script of a Bayesian network
    :return: BayesianNetwork
    """
    js_bn = bn_script_to_json(script)
    return BayesianNetwork(js_bn)


def bn_from_json(js_bn):
    """
    Build a Bayesian network from json input
    :param js_bn: json, json formatted Bayesian network
    :return: BayesianNetwork
    """
    return BayesianNetwork(js_bn)


if __name__ == '__main__':
    scr = '''
    PCore A {
        w = 1
        x1 = 1/x
        v ~ norm(z, 0.1)
        x = 0.2
        y ~ exp(x1)
        z ~ norm(w, y)
    }
    '''

    js = bn_script_to_json(scr)
    print(js)

    ex = BayesianNetwork(js)

    print('\nTo JSON, From JSON')
    print(ex)
