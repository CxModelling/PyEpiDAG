import epidag as dag
from epidag.bayesnet import Gene
from abc import ABCMeta, abstractmethod
import networkx as nx


__all__ = ['sample', 'sample_minimally', 'form_hierarchy',
           'analyse_node_type', 'formulate_blueprint', 'SimulationCore']


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


def form_hierarchy(bn, hie=None, condense=True, root='root'):
    """

    :param bn: epidag.BayesNet, a Bayesian Network
    :param hie: hierarical structure of the nodes of bn
    :param condense: True if attempting to hoist nodes to high hierarchy as possible
    :param root: name of root group
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
        root = define_node(root, hie)

    elif isinstance(hie, list):
        root = NodeGroup(root, hie[0])
        ng1 = root
        for i, hi in enumerate(hie[1:], 2):
            ng0, ng1 = ng1, NodeGroup('Layer {}'.format(i), hi)
            ng0.append_chd(ng1)
    else:
        root = NodeGroup(root, bn.OrderedNodes)

    # root.print()

    all_fixed = root.get_all()
    all_floated = [nod for nod in bn.OrderedNodes if nod not in all_fixed]

    all_floated.reverse()

    for nod in bn.ExogenousNodes:
        root.catch(nod)

    all_floated = [nod for nod in all_floated if nod not in bn.ExogenousNodes]
    for nod in all_floated:
        root.pass_down(nod, g)

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


def formulate_blueprint(bn, root, random, out):
    """
    a blueprint of a simulation model based on given a Bayesian network.
    It describes every node in the network as 1) fixed variable, 2) random variable, 3) exposed distribution
    :param bn: epidag.BayesNet, a Bayesian Network
    :param root: root node group
    :param random: nodes with random effects within an individual
    :param out: nodes can be used in simulation model
    :return: a blueprint of simulation model
    """
    suggest = analyse_node_type(bn, root, report=False)
    random = random if random else list()
    out = out if out else set.union(*[set(act) for (_, _, act) in suggest.values()])
    out = [o for o in out if o not in random]

    approved = dict()
    for k, (fs, rs, cs) in suggest.items():
        afs = list(fs)

        ars = list()
        for r in rs:
            (ars if r in random else afs).append(r)

        acs = list()
        for c in cs:
            if c in out:
                acs.append(c)
            elif c in random:
                ars.append(c)
            else:
                afs.append(c)

        approved[k] = afs, ars, acs
    # todo detect unwilling changes
    return approved


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

    def __repr__(self):
        return '{} ({})'.format(self.Field, '->'.join(f.Name for f in self.Flow))


class SingleActor(SimulationActor):
    def __init__(self, field, di):
        SimulationActor.__init__(self, field)
        self.Loci = di

    def sample(self, pas=None):
        pas = dict(pas) if pas else dict()
        return self.Loci.sample(pas)

    def __repr__(self):
        return '{} ({})'.format(self.Field, self.Loci.Func)


class FrozenSingleActor(SimulationActor):
    def __init__(self, field, di, pas):
        SimulationActor.__init__(self, field)
        self.Loci = di
        self.Dist = di.get_distribution(pas)

    def sample(self, pas=None):
        return self.Dist.sample()

    def __repr__(self):
        return '{} ({})'.format(self.Field, self.Dist.Dist)


class ParameterCore(Gene):
    def __init__(self, nickname, sg, vs, prior):
        Gene.__init__(self, vs, prior)
        self.Nickname = nickname
        self.SG = sg
        self.Parent = None
        self.Actors = dict()
        self.Children = dict()
        self.ChildrenActors = dict()

    @property
    def Group(self):
        return self.SG.Name

    def breed(self, nickname, group):
        """
        Generate an offspring node
        :param nickname: nickname
        :param group: target group of new parameter node
        :return:
        """
        if nickname in self.Children:
            raise ValueError('{} has already existed'.format(nickname))
        chd = self.SG.breed(nickname, group, self)
        self.Children[nickname] = chd
        return chd

    def list_sampler(self):
        for k in self.Actors.keys():
            yield k

        if self.Parent:
            actors = self.Parent.ChildActors[self.SG.Name]
            for k in actors:
                yield k

    def get_sampler(self, sampler):
        """
        Get a sampler of a specific variable
        :param sampler: name of the target sampler
        :return:
        """
        try:
            return self.Actors[sampler]
        except KeyError:
            pass

        try:
            return self.Parent.ChildActors[self.SG.Name][sampler]
        except AttributeError:
            raise KeyError('No {} found')
        except KeyError:
            raise KeyError('No {} found')

    def get_child(self, name):
        return self.Children[name]

    def find_child(self, address):
        """
        Find a child node
        :param address: str, a series of names of nodes linked with '@'
        :return: a child node in the address
        """
        sel = self
        names = address.split('@')
        if len(names) < 2:
            return sel
        for name in names[1:]:
            sel = sel.get_child(name)
        return sel

    def impulse(self, imp, shocked=None):
        """
        Do interventions
        :param imp: dict, intervention
        :param shocked: Do not manually input
        """
        imp = dict(imp)
        if shocked is None:
            g = self.SG.SC.BN.DAG
            shocked = set.union(*[set(nx.descendants(g, k)) for k in imp.keys()])

        shocked_locus = [s for s in shocked if s in self.Locus]
        shocked_actors = [k for k, v in self.Actors.items() if k in shocked and isinstance(v, FrozenSingleActor)]

        shocked_hoist = dict()
        for k, v in self.ChildrenActors.items():
            shocked_hoist[k] = [s for s, t in v.items() if s in shocked and isinstance(v, FrozenSingleActor)]

        if shocked_locus or shocked_actors or shocked_hoist:
            self.SG.set_response(imp, shocked_locus, shocked_actors, shocked_hoist, self)

        for v in self.Children.values():
            v.impulse(imp, shocked)

    def reset_sc(self, sc):
        self.SG = sc[self.SG.Name]
        for v in self.Children.values():
            v.reset_sc(self, sc)

    @property
    def DeepLogPrior(self):
        """
        Log prior with that of offsprings
        :return: log prior probability
        """
        return self.LogPrior + sum(v.DeepLogPrior for v in self.Children.values())

    def __iter__(self):
        if self.Parent:
            for v in iter(self.Parent):
                yield v
        for v in self.Locus.items():
            yield v

    def to_json(self):
        return {
            'Locus': self.Locus,
            'LogPrior': self.LogPrior,
            'LogLikelihood': self.LogLikelihood
        }

    def deep_print(self, i=0):
        prefix = '--' * i + ' ' if i else ''
        print('{}{} ({})'.format(prefix, self.Nickname, self))
        for k, chd in self.Children.items():
            chd.deep_print(i + 1)


class SimulationGroup:
    def __init__(self, name, fixed, random, actors, pas):
        self.Name = name
        self.SC = None
        self.Listening = list(pas)
        self.BeFixed = set(fixed)
        self.BeRandom = set(random)
        self.BeActors = set(actors)
        self.FixedChain = None
        self.Children = list()

    def set_simulation_core(self, sc):
        self.SC = sc
        bn = sc.BN
        self.FixedChain = [bn[node] for node in bn.sort(self.BeFixed)]

    def form_actor(self, bn, g, act, pas):
        pa = set(g.predecessors(act))
        if pa.intersection(self.BeRandom):
            flow = set.intersection(nx.ancestors(g, act), self.BeRandom)
            flow = bn.sort(flow)
            flow = [bn[nod] for nod in flow]
            return CompoundActor(act, flow)
        elif pa.intersection(self.BeFixed):
            return SingleActor(act, bn[act])
        else:
            return FrozenSingleActor(act, bn[act], pas)

    def actors(self, pas):
        bn = self.SC.BN
        g = bn.DAG

        for act in self.BeActors:
            yield act, self.form_actor(bn, g, act, pas)

    def generate(self, nickname, exo, actor=True):
        pas = dict(exo)
        vs = dict(pas)
        prior = 0
        for loci in self.FixedChain:
            if loci.Name not in vs:
                vs[loci.Name] = loci.sample(vs)
            prior += loci.evaluate(vs)

        vs = {k: v for k, v in vs.items() if k in self.BeFixed}

        pc = ParameterCore(nickname, self, vs, prior)
        if actor:
            pc.Actors = dict(self.actors(vs))
        return pc

    def set_response(self, imp, fixed, actors, hoist, pc):
        prior = 0
        for loci in self.FixedChain:
            if loci.Name in imp:
                pc.Locus[loci.Name] = imp[loci.Name]
            elif loci.Name in fixed:
                pc.Locus[loci.Name] = loci.sample(pc)
            prior += loci.evaluate(pc)

        pc.LogPrior = prior

        bn = self.SC.BN

        for act in actors:
            pc.Actors[act] = FrozenSingleActor(act, bn[act], pc)

        for chd, acts in hoist.items():
            for act in acts:
                pc.ChildrenActors[chd][act] = FrozenSingleActor(act, bn[act], pc)

    def breed(self, nickname, group, pa):
        if group not in self.Children:
            raise KeyError('No matched group')

        ch_sc = self.SC[group]
        if self.SC.Hoist:
            chd = ch_sc.generate(nickname, pa, False)
            chd.Parent = pa
            if group not in pa.ChildrenActors:
                pa.ChildrenActors[group] = dict(self.actors(chd))
        else:
            chd = ch_sc.generate(nickname, pa, True)
            chd.Parent = pa


        return chd

    def __repr__(self):
        return '{}({}|{}|{}->{})'.format(self.Name,
                                         self.BeFixed, self.BeRandom, self.BeActors,
                                         self.Children)

    def to_json(self):
        return {
            'Name': self.Name,
            'Listening': list(self.Listening),
            'BeFixed': list(self.BeFixed),
            'BeRandom': list(self.BeRandom),
            'BeActors': list(self.BeActors),
            'Children': list(self.Children)
        }


def get_simulation_groups(bn, bp, root):
    g = bn.DAG

    sgs = dict()
    for k, (fs, rs, cs) in bp.items():
        nodes = set(fs + rs + cs)
        pas = set.union(*[set(g.predecessors(node)) for node in nodes])
        pas = pas - nodes
        sgs[k] = SimulationGroup(k, fs, rs, cs, pas)

    def set_children(ng):
        sg = sgs[ng.Name]
        for chd in ng.Children:
            sg.Children.append(chd.Name)
            set_children(chd)

    set_children(root)

    return sgs


class SimulationCore:
    def __init__(self, bn, bp=None, root=None, hoist=True):
        self.Name = bn.Name
        self.BN = bn
        self.RootSG = root.Name
        self.SGs = get_simulation_groups(bn, bp, root)
        for sg in self.SGs.values():
            sg.set_simulation_core(self)
        self.Hoist = hoist

    def __getitem__(self, item):
        return self.SGs[item]

    def get(self, item):
        try:
            return self.SGs[item]
        except KeyError:
            raise KeyError('Unknown group')

    def generate(self, nickname, exo):
        exo = exo if exo else dict()
        return self.SGs[self.RootSG].generate(nickname, exo)

    def to_json(self):
        return {
            'BayesianNetwork': self.BN.to_json(),
            'Root': self.RootSG
        }

    def __repr__(self):
        return 'Simulation core: {}'.format(self.Name)

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
