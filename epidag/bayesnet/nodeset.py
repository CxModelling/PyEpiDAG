import epidag.bayesnet.dag as dag


__author__ = 'TimeWz'
__all__ = ['NodeSet']


class ActorBlueprint:
    Compound = 'Cp'
    Single = 'Sg'
    Frozen = 'Fz'

    def __init__(self, name, actor_type, to_read=None, to_sample=None):
        self.Name = name
        self.Type = actor_type
        self.ToRead = list(to_read) if to_read else list()
        self.ToSample = list(to_sample) if to_sample else list()

    def compose_actor(self, bn):
        if self.Type is ActorBlueprint.Frozen:
            return  # todo
        elif self.Type is ActorBlueprint.Single:
            return  # todo
        else:
            return  # todo

    def __str__(self):
        st = '{}: {}'.format(self.Type, self.Name)

        if self.ToSample:
            st = '{}, ({})'.format(st, ', '.join(self.ToSample))

        if self.ToRead:
            st = '{}| ({})'.format(st, ', '.join(self.ToRead))
        return st


class NodeSet:
    def __init__(self, name, as_fixed=None, as_floating=None):
        self.Name = name
        self.__parent = None
        self.__children = dict()
        self.__as_fixed = set(as_fixed) if as_fixed else set()
        self.__as_floating = set(as_floating) if as_floating else set()

        self.ExoNodes = None
        self.ListeningNodes = None
        self.FixedNodes = None
        self.FloatingNodes = None

        self.__was_fixed = None
        self.__will_be_floating = None
        self.Samplers = None
        self.ChildrenSamplers = None

    def add_child(self, ns):
        self.__children[ns.Name] = ns
        ns.__parent = self

    def inject_bn(self, bn):
        '''
        Allocate the nodes according to the DAG in bn
        :param bn: A well-defined BayesNet
        :return:
        '''
        assert self.validate_initial_conditions(bn)

        self.resolve_local_nodes(bn)

        self.pass_down_fixed()
        self.raise_up_floating()
        self.resolve_relations(bn)
        self.define_sampler_blueprints(bn)

    def validate_initial_conditions(self, bn):
        for ch in self.__children.values():
            if not ch.validate_initial_conditions(bn):
                return False

        g = bn.DAG
        anc = set()
        for d in self.__as_fixed:
            anc.update(g.ancestors(d))

        for d in self.__as_floating:
            if d in self.__as_fixed:
                continue
            if d in anc:
                return False
        else:
            return True

    def resolve_local_nodes(self, bn):
        g = bn.DAG
        mini = dag.minimal_dag(g, set.union(self.__as_fixed, self.__as_floating)).order()

        med = set(mini)
        med.difference_update(self.__as_fixed)
        med.difference_update(self.__as_floating)
        med = g.sort(med)

        self.FloatingNodes = set(self.__as_floating)
        self.FixedNodes = set(self.__as_fixed)

        for d in med:
            if bn.has_randomness(d, self.__as_fixed):
                self.FloatingNodes.add(d)
            else:
                self.FixedNodes.add(d)

        rqs = dict()
        self.ListeningNodes = set()  # requirements for floating nodes (giving values when needed)
        self.ExoNodes = set()  # requirements for fixed nodes (giving values at initialisation)

        for i, node in enumerate(mini):
            par = mini[:i]
            rq = dag.minimal_requirements(g, node, par)

            if node in self.FloatingNodes:
                self.ListeningNodes.update(rq)
            else:
                self.ExoNodes.update(rq)
            rqs[node] = rq
        self.ListeningNodes.difference_update(mini)
        # lis.difference_update(Fixed)
        self.ExoNodes.difference_update(self.FixedNodes)

        for ch in self.__children.values():
            ch.resolve_local_nodes(bn)

    def pass_down_fixed(self, fixed=None):
        if self.__parent:
            self.__was_fixed = fixed if fixed else {}
        else:
            self.__was_fixed = set()
        all_fixed = set.union(self.__was_fixed, self.FixedNodes)
        for ch in self.__children.values():
            ch.pass_down_fixed(all_fixed)

    def hoist_node(self, node):
        if node not in self.__was_fixed:
            self.FixedNodes.add(node)
            self.__was_fixed.add(node)

    def raise_up_floating(self):
        self.__will_be_floating = set()
        for ch in self.__children.values():
            self.__will_be_floating.update(ch.raise_up_floating())

        return set.union(self.__will_be_floating, self.FloatingNodes)

    def resolve_relations(self, bn):
        g = bn.DAG

        to_shift = set([d for d in self.ExoNodes if d not in self.__was_fixed])

        to_hoist = [d for d in to_shift if not bn.has_randomness(d, self.__was_fixed)]
        to_shift.difference_update(to_hoist)

        self.ExoNodes.difference_update(to_shift)
        self.FixedNodes.update(to_shift)

        for d in to_hoist:
            self.__parent.hoist_node(d)

        for ch in self.__children.values():
            ch.resolve_relations(bn)

    def define_sampler_blueprints(self, bn):
        g = bn.DAG
        self.Samplers = dict()
        self.ChildrenSamplers = {ch: dict() for ch in self.__children.keys()}

        af = set.union(self.__was_fixed, self.FixedNodes)

        for d in set.union(self.FloatingNodes, self.__will_be_floating):
            loci = bn[d]
            pars = set(loci.Parents)

            if pars < af:
                self.Samplers[d] = ActorBlueprint(d, ActorBlueprint.Frozen, pars, None)

                for k, ch in self.__children.items():
                    if d in ch.FloatingNodes:
                        if set.intersection(pars, ch.FixedNodes):
                            self.ChildrenSamplers[k][d] = ActorBlueprint(d, ActorBlueprint.Single,
                                                                         pars, None)
                        else:
                            self.ChildrenSamplers[k][d] = None
                    elif d in ch.__will_be_floating:
                        self.ChildrenSamplers[k][d] = None
            else:
                req = dag.minimal_requirements(g, d, af)
                to_read = af.intersection(req)
                to_sample = [n for n in req if n not in to_read]
                self.Samplers[d] = ActorBlueprint(d, ActorBlueprint.Compound, to_read, to_sample)

                for k, ch in self.__children.items():
                    if d in ch.__will_be_floating:
                        self.ChildrenSamplers[k][d] = None

        for ch in self.__children.values():
            ch.define_sampler_blueprints(bn)

    def print_samplers(self, i=0):
        ind = i * ' '
        print('{}NodeSet {}'.format((i - 2) * ' ' + '|-' if i else '', self.Name))
        for bp in self.Samplers.values():
            print('{}|-{}'.format(ind, bp))

        for ch in self.__children.values():
            ch.print_samplers(i + 2)

    def has_floating_blueprint(self, node):
        return node in self.Samplers

    def print(self, i=0):
        ind = i * ' '
        print('{}NodeSet {}'.format((i - 2) * ' ' + '|-' if i else '', self.Name))
        print('{}|-Fixed {}'.format(ind, self.FixedNodes))
        print('{}|-Floating {}'.format(ind, self.FloatingNodes))
        print('{}|-Exo {}'.format(ind, self.ExoNodes))
        print('{}|-Listening {}'.format(ind, self.ListeningNodes))
        for ch in self.__children.values():
            ch.print(i + 2)


if __name__ == '__main__':
    from epidag.bayesnet.bn import bayes_net_from_script

    bn = bayes_net_from_script('''
    PCore Test {
        a = 1
        b = a + 3
        c = b * 2
        d ~ binom(b, 0. 5)
        e = d + c
    }
    '''
                                   )

    nr = NodeSet('root', as_fixed=['a'])
    ns = NodeSet('med', as_fixed=['b'], as_floating=['d'])
    ne = NodeSet('leaf', as_fixed=['e'])
    nr.add_child(ns)
    ns.add_child(ne)

    nr.inject_bn(bn)
    nr.print()

    nr.print_samplers()
