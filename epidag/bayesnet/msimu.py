import epidag as dag
from abc import ABCMeta, abstractmethod
import networkx as nx


__all__ = ['sample', 'sample_minimally', 'form_hierarchy',
           'analyse_node_type', 'to_simulation_core',
           'CompoundActor', 'SingleActor', 'FrozenSingleActor']


def sample(bn, cond=None):
    g = bn.DAG
    cond = cond if cond else dict()
    if any(nod not in cond for nod in bn.ExogenousNodes):
        raise ValueError('Exogenous nodes do not fully defined')

    res = dict(cond)

    for nod in bn.OrderedNodes:
        if nod not in res:
            res[nod] = g.nodes[nod]['loci'].sample(res)
    return res


def sample_minimally(bn, included, cond, sources=False):
    """
    sample variables which are minimal requirements of having included
    :param bn: a Bayesian Network
    :param included: iterable, targeted output variables
    :param cond: dict, given variables
    :param sources: True if mediators needed
    :return:
    """
    g = bn.DAG

    cond = cond if cond else dict()
    given = list(cond.keys())

    suf = dag.get_sufficient_nodes(g, included, given)
    suf_exo = [nod for nod in bn.ExogenousNodes if nod in suf]

    for nod in suf_exo:
        if nod not in cond:
            raise ValueError('Exogenous node {} does not found'.format(nod))

    res = dict(cond)

    for nod in bn.OrderedNodes:
        if nod in suf and nod not in res:
            res[nod] = g.nodes[nod]['loci'].sample(res)
    sinks = {k: v for k, v in res.items() if k in included}
    if sources:
        med = {k: v for k, v in res.items() if k not in included}
        return sinks, med
    else:
        return sinks


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


def form_hierarchy(bn, hie=None, condense=True):
    """

    :param bn: epidag.BayesNet, a Bayesian Network
    :param hie: hierarical structure of the nodes of bn
    :param condense: True if attempting to hoist nodes to high hierarchy as possible
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
        root = define_node('root', hie)

    elif isinstance(hie, list):
        root = NodeGroup('root', hie[0])
        ng1 = root
        for i, hi in enumerate(hie[1:], 2):
            ng0, ng1 = ng1, NodeGroup('Layer {}'.format(i), hi)
            ng0.append_chd(ng1)
    else:
        root = NodeGroup('root', bn.OrderedNodes)

    # root.print()

    all_fixed = root.get_all()
    all_floated = [nod for nod in bn.OrderedNodes if nod not in all_fixed]

    all_floated.reverse()

    for nod in bn.ExogenousNodes:
        root.catch(nod)

    all_floated = [nod for nod in all_floated if nod not in bn.ExogenousNodes]
    for nod in all_floated:
        root.pass_down(nod, g)

    if not condense:
        return root

    all_floated.reverse()

    for nod in all_floated:
        root.raise_up(nod, g)

    return root


def analyse_node_type(bn, root, report=False):
    """
    Analyse nodes in each group based on their potential characteristics.
    A node which can be an actor must be a leaf of the given DAG.
    A node which can carry stochastic effects must not be an ancestor of nodes in lower models.

    :param bn: epidag.BayesNet, a Bayesian Network
    :param root: root node group
    :param report: True if report print needed
    :return: dict, key = Group name, value = (Should be fixed, Can be random, Can be actors)
    """
    g = bn.DAG
    leaves = bn.LeafNodes

    res = dict()

    def fn(ng, ind=0):
        fix, ran, act = list(), list(), list()
        for node in ng.Nodes:
            if node in leaves:
                act.append(node)
            elif nx.descendants(g, node) < ng.Nodes:
                ran.append(node)
            else:
                fix.append(node)

        res[ng.Name] = fix, ran, act
        if report:
            print('{}Group {}'.format('--' * ind, ng.Name))
            print('{}Must be fixed: {}'.format('  ' * ind, fix))
            print('{}Can be random: {}'.format('  ' * ind, ran))
            print('{}Can be actors: {}'.format('  ' * ind, act))

        for chd in ng.Children:
            fn(chd, ind + 1)

    fn(root)
    return res


class SimulationGroup:
    def __init__(self, bn, ng, fixed, random, actors, pas):
        self.Name = ng.Name
        self.BN = bn
        self.Listening = set(pas)
        self.BeFixed = set(fixed)
        self.BeRandom = set(random)
        self.BeActors = set(actors)
        self.Children = dict()

    def sample(self, nickname):
        pass

    def to_json(self):
        return {
            'Name': self.Name,
            'Listening': list(self.Listening),
            'BeFixed': list(self.BeFixed),
            'BeRandom': list(self.BeRandom),
            'BeActors': list(self.BeActors),
            'Children': {k: v.to_json() for k, v in self.Children.items()}
        }

    def copy(self):
        pass


def to_simulation_core(bn, hie=None, bp=None, root=None, reduce=True):
    if not bp:
        root = root if root else dag.form_hierarchy(bn, hie)
        bp = dag.analyse_node_type(bn, root)

    g = bn.DAG

    if reduce:
        bp = {k: (v0 + v1, [], v2) for k, (v0, v1, v2) in bp.items()}

    def fn(ng):
        fi, ra, ac = bp[ng.Name]
        pas = set.union(*[set(g.predecessors(node)) for node in ng.Nodes])
        pas = pas - ng.Nodes
        sg = SimulationGroup(bn, ng, fi, ra, ac, pas)

        for chd in ng.Children:
            sg.Children[chd.Name] = fn(chd)

        return sg

    return fn(root)



class SimulationActor(metaclass=ABCMeta):
    def __init__(self, field):
        self.Field = field

    @abstractmethod
    def sample(self, pas=None):
        pass


class CompoundActor(SimulationActor):
    def __init__(self, field, flow):
        SimulationActor.__init__(self, field)
        self.Flow = list(flow)

    def sample(self, pas=None):
        pas = dict(pas) if pas else dict()
        for loc in self.Flow:
            pas[loc.Name] = loc.sample(pas)
        return pas[self.Field]


class SingleActor(SimulationActor):
    def __init__(self, field, di):
        SimulationActor.__init__(self, field)
        self.Loci = di

    def sample(self, pas=None):
        pas = dict(pas) if pas else dict()
        return self.Loci.sample(pas)


class FrozenSingleActor(SimulationActor):
    def __init__(self, field, di, pas):
        SimulationActor.__init__(self, field)
        self.Loci = di
        self.Dist = di.get_distribution(pas)

    def sample(self, pas=None):
        return self.Dist.sample()


'''
class ParameterCore(Gene):
    def __init__(self, ds, vs):
        Gene.__init__(self, vs)
        self.Distributions = dict(ds)

    def get_distribution(self, d):
        return self.Distributions[d]

    def clone(self):
        g = ParameterCore(self.Distributions, self.Locus)
        g.LogPrior = self.LogPrior
        g.LogLikelihood = self.LogLikelihood
        return g

    def difference(self, other):
        hyper = list()
        leaves = list()

        for k, v in self.Locus.items():
            if k in other.Locus:
                if other.Locus[k] != v:
                    hyper.append(k)

        for k, v in self.Distributions.items():
            if k in other.Distributions:
                if str(other.Distributions[k]) != str(v):
                    leaves.append(k)

        return hyper, leaves

    def __getitem__(self, item):
        return self.Distributions[item]

    def __contains__(self, item):
        return item in self.Distributions

    def get(self, item):
        try:
            return self.Locus[item]
        except KeyError:
            try:
                return self.Distributions[item].sample()
            except KeyError as k:
                raise k

    def __repr__(self):
        s = Gene.__repr__(self) + ', '
        s += ", ".join(['{}~{}'.format(k, v) for k, v in self.Distributions.items()])
        return s


class SimulationModel:
    def __init__(self, model):
        self.DAG = model

    @property
    def Name(self):
        return self.DAG.Name

    def sample_core(self, cond=None):
        """
        Sample a parameter core with prior probability
        :return: ParemeterCore: a prior parameter core
        """
        ds, vs = self.DAG.sample_leaves(need_vs=True, cond=cond)
        g = ParameterCore(ds, vs)
        g.LogPrior = self.DAG.evaluate(g)
        return g

    def mutate(self, pcs):
        """
        jitter the value of parameterCore
        :param pcs: list<ParameterCore>: original parameter cores
        :return: List<ParameterCore>: mutated pcs
        """
        dat = pd.DataFrame.from_records([pc.Locus for pc in pcs])
        ds = self.DAG.sample_distributions()

        for k in dat:
            try:
                di = ds[k]
                if di.Type is 'Double':
                    amt = dat[k]
                    amt = 0.01 * (amt.max() - amt.min())
                    amt = rd.normal(dat[k], scale=amt, size=len(dat[k]))
                    amt = np.minimum(np.maximum(amt, di.Lower), di.Upper)
                    dat[k] = amt
            except KeyError:
                continue

        return [self.reform_core(locus) for k, locus in dat.iterrows()]

    def intervene_core(self, pc, intervention):
        ds, vs = self.DAG.intervene_leaves(intervention, pc.Locus)
        g = ParameterCore(ds, vs)
        g.LogPrior = self.DAG.evaluate(g)
        return g

    def reform_core(self, vs):
        """
        Use new table to generate a parameter core
        :param vs: parameter table
        :return: ParameterCore: new core
        """
        vs = dict(vs)
        ds = self.DAG.sample_leaves(vs)
        g = ParameterCore(ds, vs)
        self.DAG.regularise(g)
        return g

    def __str__(self):
        return str(self.DAG)

    __repr__ = __str__

    def to_json(self):
        return self.DAG.to_json()
'''
