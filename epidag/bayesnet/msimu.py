import epidag as dag
import networkx as nx


__all__ = ['sample', 'sample_minimally', 'form_hierarchy']


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
    def __init__(self, fixed):
        self.Children = set()
        self.Nodes = set(fixed)
        self.Parents = set()

    def append_chd(self, chd):
        self.Children.add(chd)

    def needs(self, nod, g):
        if any(nod in nx.ancestors(g, x) for x in self.Nodes):
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
        if not self.can_be_passed_down(nod, g):
            return

        needed = [chd for chd in self.Children if chd.needs(nod, g)]
        if len(needed) is 1:
            self.Nodes.remove(nod)
            needed[0].catch(nod)
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
                chd.Nodes.remove(nod)
                self.catch(nod)
                return

    def get_all(self):
        return set.union(self.Nodes, *[chd.get_all() for chd in self.Children])

    def print(self, i=0):
        print('--' * i + '(' + ', '.join(self.Nodes) + ')')
        for chd in self.Children:
            chd.print(i + 1)


def form_hierarchy(bn, hie=None, condense=True):
    g = bn.DAG

    # todo tree form
    # check order

    if not hie:
        root = NodeGroup(bn.OrderedNodes)

    if isinstance(hie, list):
        root = NodeGroup(hie[0])
        ng1 = root
        for hi in hie[1:]:
            ng0, ng1 = ng1, NodeGroup(hi)
            ng0.append_chd(ng1)

    else:
        root = NodeGroup(bn.OrderedNodes)
    # elif isinstance(hie, dict):
    #    # todo
    #    pass

    all_fixed = root.get_all()
    all_floated = [nod for nod in bn.OrderedNodes if nod not in all_fixed]

    all_floated.reverse()

    for nod in bn.ExogenousNodes:
        root.catch(nod)

    all_floated = [nod for nod in all_floated if nod not in bn.ExogenousNodes]
    for nod in all_floated:
        root.catch(nod)
        root.pass_down(nod, g)

    if not condense:
        return root

    all_floated.reverse()

    for nod in all_floated:
        root.raise_up(nod, g)

    return root


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

if __name__ == '__main__':
    pars1 = """
    {
        x = 1
        beta0 ~ norm(10, 0.02)
        beta1 ~ norm(0.5, 0.1)
        mu = beta0 + beta1*x
        sigma ~ gamma(0.01, 0.01)
        y ~ norm(mu, sigma)
    }
    """

    from epidag import DirectedAcyclicGraph
    dag1 = DirectedAcyclicGraph(pars1)

    sim = dag1.get_simulation_model()
    print(sim)

    pc1 = sim.sample_core()
    print(pc1)
    pc2 = sim.sample_core()
    print(pc2)
    print(pc1.difference(pc2))
    print(dag1.get_descendants(['x']))

    pars2 = """
        {
            a = 1
            b ~ unif(1, 4)
            c = a + b
            d ~ k(b + c)
            e ~ exp(b)
        }
        """

    dag2 = DirectedAcyclicGraph(pars2)

    sim = dag2.get_simulation_model()
    print(sim)

    pc1 = sim.sample_core()
    print(pc1)
    pc2 = sim.intervene_core(pc1, {'a': 3})
    print(pc2)
