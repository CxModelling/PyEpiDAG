import epidag.bayesnet.dag as dag
import epidag.simulation.actor as act

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
            return  act.FrozenSingleActor(self.Name, bn[self.Name], self.ToRead)
        elif self.Type is ActorBlueprint.Single:
            return  act.SingleActor(self.Name, bn[self.Name], self.ToRead)
        else:
            to_sample = [bn[d] for d in self.ToSample]
            return  act.CompoundActor(self.Name, bn[self.Name], self.ToRead, to_sample)

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
        self.LocalSamplers = None
        self.SharedSamplers = None
        self.__frozen = False

    def defrost(self):
        self.ExoNodes = None
        self.ListeningNodes = None
        self.FixedNodes = None
        self.FloatingNodes = None

        self.__was_fixed = None
        self.__will_be_floating = None
        self.Samplers = None
        self.ChildrenSamplers = None
        self.__frozen = False


    def add_child(self, ns):
        assert not self.__frozen
        self.__children[ns.Name] = ns
        ns.__parent = self

    def new_child(self, name, as_fixed=None, as_floating=None):
        ns = NodeSet(name, as_fixed, as_floating)
        self.add_child(ns)
        return ns

    @property
    def Children(self):
        return self.__children

    @property
    def Parent(self):
        return self.Parent

    def inject_bn(self, bn):
        '''
        Allocate the nodes according to the DAG in bn
        :param bn: A well-defined BayesNet
        :return:
        '''
        assert not self.__frozen
        assert self._validate_initial_conditions(bn)

        self._resolve_local_nodes(bn)

        self._pass_down_fixed()
        self._raise_up_floating()
        self._resolve_relations(bn)
        self._define_sampler_blueprints(bn)
        self._sort_fixed_nodes(bn)
        self.__frozen = True

    def _validate_initial_conditions(self, bn):
        for ch in self.__children.values():
            if not ch._validate_initial_conditions(bn):
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

    def _resolve_local_nodes(self, bn):
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

        for d in bn.sort(self.FloatingNodes):
            if not bn.has_randomness(d, self.FixedNodes):
                self.FixedNodes.add(d)
                self.FloatingNodes.remove(d)


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
            ch._resolve_local_nodes(bn)

    def _pass_down_fixed(self, fixed=None):
        if self.__parent:
            self.__was_fixed = fixed if fixed else set()
        else:
            self.__was_fixed = set()
        all_fixed = set.union(self.__was_fixed, self.FixedNodes)
        for ch in self.__children.values():
            ch._pass_down_fixed(all_fixed)

    def _hoist_node(self, node):
        if node not in self.__was_fixed:
            self.FixedNodes.add(node)
            self.__was_fixed.add(node)

    def _raise_up_floating(self):
        self.__will_be_floating = set()
        for ch in self.__children.values():
            self.__will_be_floating.update(ch._raise_up_floating())

        return set.union(self.__will_be_floating, self.FloatingNodes)

    def _resolve_relations(self, bn):
        to_shift = set([d for d in self.ExoNodes if d not in self.__was_fixed])

        to_hoist = [d for d in to_shift if not bn.has_randomness(d, self.__was_fixed)]
        to_shift.difference_update(to_hoist)

        self.ExoNodes.difference_update(to_shift)
        self.FixedNodes.update(to_shift)

        for d in to_hoist:
            self.__parent._hoist_node(d)


        to_shift = [d for d in self.ListeningNodes if d not in self.__was_fixed]
        to_shift = [d for d in to_shift if not bn.is_exogenous(d)]
        to_shift = [d for d in to_shift if bn.is_deterministic(d, self.__was_fixed)]
        self.ListeningNodes.difference_update(to_shift)
        self.FixedNodes.update(to_shift)


        for ch in self.__children.values():
            ch._resolve_relations(bn)

    def _define_sampler_blueprints(self, bn):
        g = bn.DAG
        self.LocalSamplers = dict()
        self.SharedSamplers = dict()

        af = set.union(self.__was_fixed, self.FixedNodes)

        # If all parent nodes have been fixed without local changes -> f, f
        # If all parent nodes have been fixed but have local changes -> s, f
        # If any parent nodes is still floating -> c, c
        for d in set.union(self.FloatingNodes, self.__will_be_floating):
            loci = bn[d]
            pars = set(loci.Parents)

            if pars <= af: # if all parent nodes had been fixed before
                self.LocalSamplers[d] = ActorBlueprint(d, ActorBlueprint.Frozen, pars, None)
                if set.intersection(pars, self.FixedNodes):

                    self.SharedSamplers[d] = ActorBlueprint(d, ActorBlueprint.Single, pars, None)
                else:
                    self.SharedSamplers[d] = self.LocalSamplers[d]
            else:
                req = dag.minimal_requirements(g, d, af)
                to_read = [n for n in req if n in af or bn.is_exogenous(n)]
                to_sample = [n for n in req if n not in to_read]
                actor = ActorBlueprint(d, ActorBlueprint.Compound, to_read, to_sample)
                self.LocalSamplers[d] = self.SharedSamplers[d] = actor

        for ch in self.__children.values():
            ch._define_sampler_blueprints(bn)

    def _sort_fixed_nodes(self, bn):
        self.FixedNodes = bn.sort(self.FixedNodes)
        for ch in self.__children.values():
            ch._sort_fixed_nodes(bn)

    def print_samplers(self, i=0):
        ind = i * ' '
        print('{}NodeSet {}'.format((i - 2) * ' ' + '|-' if i else '', self.Name))
        for bp in self.LocalSamplers.values():
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
    ns = nr.new_child('med', as_fixed=['b'], as_floating=['d'])
    ne = ns.new_child('leaf', as_fixed=['e'])

    nr.inject_bn(bn)

    nr.print()