import epidag as dag
import json
import re
import networkx as nx

__author__ = 'TimeWz667'


def bn_script_to_json(script):
    # remove space
    pars = script.replace(' ', '')
    pars = pars.replace('\t', '')
    # split lines
    pars = pars.split('\n')
    pars = [par for par in pars if par != '']

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
            nodes[p_name] = {'Type': 'Distribution', 'Def': p.group(2), 'Parents': pas}
        elif p.find('=') >= 0:
            p = re.match(r'(\w+)\=(\S+)', p, re.IGNORECASE)
            p_name, p_func = p.group(1), p.group(2)
            pas, fu = dag.parse_parents(p_func)
            all_fu = all_fu.union(fu)
            all_pa = all_pa.union(pas)
            if len(pas):
                node = {'Type': 'Function', 'Def': p_func, 'Parents': pas}
            else:
                node = {'Type': 'Value', 'Def': p_func}
            nodes[p_name] = node

    for pa in all_pa:
        if pa not in nodes:
            nodes[pa] = {'Type': 'ExoValue'}

    js = {'Name': name, 'Nodes': nodes, 'Dependency': all_fu}
    return js


class BayesianNetwork:
    def __init__(self, js):
        self.Name = js['Name']
        self.Source = js
        self.DAG = nx.DiGraph()

        for k, v in js['Nodes'].items():
            if v['Type'] is 'Value':
                loci = dag.ValueLoci(k, v['Def'])
            elif v['Type'] is 'ExoValue':
                loci = dag.ExoValueLoci(k)
            elif v['Type'] is 'Distribution':
                loci = dag.DistributionLoci(k, v['Def'])
            else:
                loci = dag.FunctionLoci(k, v['Def'])

            self.DAG.add_node(k, loci=loci, **v)
            if 'Parents' not in v:
                continue
            for pa in v['Parents']:
                self.DAG.add_edge(pa, k)

        if not nx.is_directed_acyclic_graph(self.DAG):
            raise SyntaxError('Cyclic groups found')
        if js['Dependency'] > dag.MATH_FUNC.keys():
            raise SyntaxError('Known functions found')

        nx.freeze(self.DAG)
        self.ExogenousNodes = [k for k, v in self.DAG.nodes.data() if v['Type'] is 'ExoValue']
        self.RootNodes = [k for k, v in self.DAG.in_degree() if v == 0]
        self.LeafNodes = [k for k, v in self.DAG.out_degree() if v == 0]
        self.OrderedNodes = self.__find_order()

    def __find_order(self):
        ordered = list()
        to_test = set(self.DAG.nodes())
        tested = set(ordered)
        while to_test:
            for nod in to_test:
                if nx.ancestors(self.DAG, nod) <= tested:
                    ordered.append(nod)
            tested = set(ordered)
            to_test = to_test - tested
        return ordered

    def to_json(self):
        return self.Source

    def __str__(self):
        ss = ['Name:\t{}'.format(self.Source['Name']), 'Nodes:']
        for k in self.OrderedNodes:
            v = self.DAG.nodes[k]
            ss.append('\t{}'.format(v['loci']))
        return '\n'.join(ss)

    __repr__ = __str__


if __name__ == '__main__':
    scr1 = '''
    PCore A {
        w = 1
        x1 = 1/x
        v ~ norm(z, 0.1)
        x = 0.2
        y ~ exp(x1)
        z ~ norm(w, y)
    }
    '''

    js1 = bn_script_to_json(scr1)
    print(js1)

    dag1 = BayesianNetwork(js1)


    print('\nTo JSON, FROM JSON')

    print(dag1)

    #print(dag1.get_offsprings('y'))

    #sp2 = dag1.shock(sp1, 'y', 10)
    #print(sp2)

