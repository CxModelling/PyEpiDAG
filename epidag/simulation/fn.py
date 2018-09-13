import epidag as dag
from epidag.simulation.simucore import SimulationCore
__author__ = 'TimeWz667'
__all__ = ['as_simulation_core', 'quick_build_parameter_core']


def as_simulation_core(bn, hie=None, root=None, random=None, out=None):
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

    ng = dag.form_hierarchy(bn, hie, root)
    bp = dag.formulate_blueprint(bn, ng, random, out)
    return SimulationCore(bn, bp, ng)


def quick_build_parameter_core(script):
    bn = dag.bayes_net_from_script(script)
    sm = as_simulation_core(bn)
    return sm.generate('')
