import epidag as dag
from epidag.simulation.nodeset import NodeSet
from epidag.simulation.simucore import SimulationCore
__author__ = 'TimeWz667'
__all__ = ['as_simulation_core', 'quick_build_parameter_core']


def as_simulation_core(bn, ns: NodeSet=None):
    """
    a blueprint of a simulation model based on given a Bayesian network.
    It describes every node in the network as 1) fixed variable, 2) random variable, 3) exposed distribution
    :param bn: epidag.BayesNet, a Bayesian Network
    :param ns: name of root group
    :return: a simulation model
    """
    if not ns:
        ns = NodeSet('Root', as_floating=bn.DAG.leaves())
    ns.inject_bn(bn)
    return SimulationCore(bn, ns)


def quick_build_parameter_core(script):
    bn = dag.bayes_net_from_script(script)
    flt = [d for d in bn.Leaves if bn.is_rv(d)]

    ns = NodeSet('Root', as_floating=flt)
    sm = as_simulation_core(bn, ns)
    return sm.generate('')
