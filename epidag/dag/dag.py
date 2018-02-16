from epidag.loci import *
from epidag import Gene, BayesianModel, SimulationModel
from collections import OrderedDict
import json
import re
from functools import reduce

__author__ = 'TimeWz667'


class ScriptException(Exception):
    def __init__(self, err):
        self.Err = err

    def __repr__(self):
        return self.Err


def script_to_json(script):
    def find_parent(expr):
        va, pa = None, list()
        while not va:
            try:
                va = eval(expr)
            except NameError as e:
                los = re.match(r"name '(\w+)'", e.args[0])
                los = los.group(1)
                exec('{} = 0.87'.format(los))  # todo potential error
                pa.append(los)
            except SyntaxError as e:
                raise e
        return pa

    pars = script.replace(' ', '')
    pars = pars.replace('\t', '')
    pars = pars.split('\n')
    pars = [par for par in pars if par != '']
    nodes = dict()
    try:
        name = re.match(r"PCore\s*(?P<name>\w+)\s*\{", pars[0], re.IGNORECASE).group('name')
    except AttributeError:
        name = 'PCore_{}'.format(1)

    for p in pars:
        if p.find('=') >= 0:
            p = p.split('=', 1)
            p_name, p_func = p[0], p[1]
            try:
                p_func = eval(p_func)
                node = {'Type': 'Value', 'Def': p_func}
            except NameError:
                pas = find_parent(p_func)
                node = {'Type': 'Function', 'Def': p_func, 'Parents': pas}
            finally:
                nodes[p_name] = node
        elif p.find('~') >= 0:
            p = p.split('~', 1)
            p_name, p_func = p[0], p[1]
            args = p_func.split('(', 1)[1][:-1]
            args = args.split(',')
            pas = reduce(lambda x, y: x + y, [find_parent(arg) for arg in args])
            pas = list(set(pas))
            nodes[p_name] = {'Type': 'Distribution', 'Def': p_func, 'Parents': pas}

    odn = OrderedDict()
    dep = 0
    while len(odn) < len(nodes) or dep > 10:
        tmp = dict()
        for k, node in nodes.items():
            if k in odn:
                continue
            if 'Parents' not in node:
                tmp[k] = node
            elif all([pa in odn for pa in node['Parents']]):
                tmp[k] = node
        for k, v in tmp.items():
            odn[k] = v
        dep += 1

    js = {'Name': name, 'Nodes': nodes, 'Order': list(odn.keys()), 'Depth': dep}
    return js


class DirectedAcyclicGraph:
    NumberDAG = 0

    def __init__(self, script, is_js=False):
        DirectedAcyclicGraph.NumberDAG += 1
        js = script if is_js else script_to_json(script)

        self.Name = js['Name']
        self.Locus = DirectedAcyclicGraph.restore_locus(js['Nodes'], js['Order'])
        self.Depth = js['Depth']
        self.Leaves = [k for k in self.Locus.keys() if not self.get_children(k)]
        self.Roots = [k for k in self.Locus.keys() if not self.get_parents(k)]
        self.Pathways = None

    def __getitem__(self, item):
        return self.Locus[item]

    def get_parents(self, node):
        return self[node].Parents

    def get_children(self, node):
        return [k for k, v in self.Locus.items() if node in v.Parents]

    def get_descendants(self, nodes):
        paths = self.Pathways if self.Pathways else self.get_pathways()

        des = list()
        for k, vs in paths.items():
            if k in nodes:
                for v in vs:
                    des += v
        des = list(set(des))
        des = [k for k in des if k not in nodes]
        return des

    def get_pathways(self):
        if self.Pathways:
            return self.Pathways
        ch = OrderedDict()
        for k in self.Locus.keys():
            ch[k] = list(self.get_children(k))

        paths = OrderedDict()
        for k in self.Locus.keys():
            ps = list()
            temp0, temp1 = [[k]], list()
            while len(temp0) > 0:
                for path in temp0:
                    cs = ch[path[-1]]
                    nc = len(cs)
                    if nc == 0:
                        ps.append(path)
                    else:
                        for c in cs:
                            p = path.copy()
                            p.append(c)
                            temp1.append(p)
                temp0, temp1 = temp1, list()
            paths[k] = ps
        self.Pathways = paths
        return paths

    def group(self, evi):
        pws = self.get_pathways()
        pws = {k: [p for p in ps if any(e in p for e in evi)] for k, ps in pws.items()}
        pws = {k: ps for k, ps in pws.items() if ps}

        group = OrderedDict()
        med = list()
        for k, paths in pws.items():
            if k not in med and k not in evi:
                group[k] = 'Prior', self.Locus[k]
            else:
                for path in paths:
                    r = max([path.index(e) for e in evi if e in path])
                    for i in range(r):
                        p = path[i]
                        if p not in evi and p not in med:
                            med.append(p)
            if k in med:
                group[k] = 'Mediator', self.Locus[k]
            elif k in evi:
                group[k] = 'Evidence', self.Locus[k]
        return group

    def sample(self, cond=None):
        vs = OrderedDict()
        cond = cond if cond else dict()
        prior = 0
        for k, node in self.Locus.items():
            if k in cond:
                vs[k] = cond[k]
                prior += node.evaluate(vs)
            else:
                try:
                    di = node.get_distribution(vs)
                    v = node.sample(vs)
                    prior += di.logpdf(v)
                except AttributeError:
                    v = node.sample(vs)
                finally:
                    vs[k] = v

        return Gene(vs, prior)

    def sample_leaves(self, cond=None, need_vs=False):
        cond = cond if cond else {}
        vs = OrderedDict()
        ds = OrderedDict()

        for loci, v in self.Locus.items():
            if loci in self.Leaves:
                try:
                    ds[loci] = v.get_distribution(vs)
                except AttributeError:
                    continue
            elif loci in cond:
                vs[loci] = cond[loci]
            else:
                vs[loci] = v.sample(vs)
        if need_vs:
            return ds, vs
        else:
            return ds

    def intervene_leaves(self, intervention, cond=None):

        cond = cond if cond else {}
        des = self.get_descendants(intervention.keys())
        cond = {k: v for k, v in cond.items() if k not in des}
        cond.update(intervention)
        return self.sample_leaves(cond, True)

    def sample_distributions(self):
        vs, ds = dict(), dict()
        for k, node in self.Locus.items():
            try:
                di = node.get_distribution(vs)
                v = node.sample(vs)
                ds[k] = di
            except AttributeError:
                v = node.sample(vs)
            finally:
                vs[k] = v

        return ds

    def evaluate(self, gene):
        """
        calculate the prior probability of a parameter table
        :param gene: a parameter table to be evaluate
        :return: prior probability in log
        """
        p = [loci.evaluate(gene.Locus) for k, loci in self.Locus.items() if k in gene.Locus]
        return sum(p)

    def regularise(self, gene):
        loc = gene.Locus
        for k, loci in self.Locus:
            if isinstance(loci, FunctionLoci) and k in loc:
                loc[k] = loci.sample(loc)
        gene.LogPrior = self.evaluate(gene)

    def is_distribution(self, node):
        return isinstance(self[node], DistributionLoci)

    def is_function(self, node):
        return isinstance(self[node], FunctionLoci)

    def is_value(self, node):
        return isinstance(self[node], ValueLoci)

    def get_bayesian_model(self, evi):
        return BayesianModel(self, self.group(evi))

    def get_simulation_model(self):
        return SimulationModel(self)

    def get_causal_diagram(self, invisible):
        # todo
        pass

    def __str__(self):
        s = 'PCore {} '.format(self.Name)
        s += '{\n\t'
        s += '\n\t'.join([str(v) for v in self.Locus.values()])
        s += '\n}'
        return s

    def to_json(self):
        return {'Name': self.Name,
                'Nodes': {k: loci.to_json() for k, loci in self.Locus.items()},
                'Order': list(self.Locus.keys()),
                'Depth': self.Depth}

    @staticmethod
    def from_json(js):
        if isinstance(js, str):
            js = json.loads(js)

        return DirectedAcyclicGraph(js, True)

    @staticmethod
    def restore_locus(nodes, order):
        locus = OrderedDict()
        for nd in order:
            node = nodes[nd]
            if node['Type'] == 'Value':
                locus[nd] = ValueLoci(nd, node['Def'])
            elif node['Type'] == 'Function':
                locus[nd] = FunctionLoci(nd, node['Def'], node['Parents'])
            elif node['Type'] == 'Distribution':
                locus[nd] = DistributionLoci(nd, node['Def'], node['Parents'])

        return locus


if __name__ == '__main__':
    scr1 = '''
    {
        w = 1
        x1 = 1/x
        v ~ norm(z, 0.1)
        x = 0.2
        y ~ exp(x1)
        z ~ norm(w, y)
    }
    '''

    js1 = script_to_json(scr1)
    print(js1)

    dag1 = DirectedAcyclicGraph(scr1)

    print(dag1)
    print('Sampling')

    sp1 = dag1.sample()
    print(sp1)

    print('\nPrior probability')
    print(dag1.evaluate(sp1))

    print('\nTo JSON, FROM JSON')
    js1 = dag1.to_json()
    print(js1)
    print(DirectedAcyclicGraph.from_json(js1))
    print(DirectedAcyclicGraph.from_json(json.dumps(js1)))
    sp1 = dag1.sample()
    print(sp1)

    #print(dag1.get_offsprings('y'))

    #sp2 = dag1.shock(sp1, 'y', 10)
    #print(sp2)

