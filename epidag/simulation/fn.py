import epidag as dag
from epidag.simulation.nodeset import NodeSet
from epidag.simulation.simucore import SimulationCore
__author__ = 'TimeWz667'
__all__ = ['as_simulation_core', 'quick_build_parameter_core']


def as_simulation_core(bn, ns: NodeSet):
    """
    a blueprint of a simulation model based on given a Bayesian network.
    It describes every node in the network as 1) fixed variable, 2) random variable, 3) exposed distribution
    :param bn: epidag.BayesNet, a Bayesian Network
    :param hie: hierarchical structure of the nodes of bn
    :param root: name of root group
    :param random: nodes with random effects within an individual
    :param out: nodes can be used in simulation model
    :return: a simulation model
    """
    ns.inject_bn(bn)
    return SimulationCore(bn, ns)


def quick_build_parameter_core(script):
    bn = dag.bayes_net_from_script(script)
    sm = as_simulation_core(bn)
    return sm.generate('')
