import epidag as dag
from .databm import as_bayesian_model_frame, get_data_shadow, \
    get_data_hierarchy, DataNodeSet, DataBayesianModel
from .simubm import SimulationBayesianModel


__author__ = 'TimeWz667'
__all__ = ['as_data_model', 'as_simulation_data_model']


def as_data_model(bn, data, exo=None, latent=None, datum_name='entries'):
    data_reformed = as_bayesian_model_frame(data, exo=exo, datum_name=datum_name)
    data_shadow = get_data_shadow(data_reformed, bn)
    data_hie = get_data_hierarchy(data_shadow, bn, latent=latent)

    ng = dag.form_hierarchy(bn, data_hie)

    root_nodes = bn.sort(ng.Nodes)
    # leaf_nodes = bn.sort(list(ng.Children)[0].Nodes)

    leaves = list()

    for datum, nodes in zip(data_reformed['entries'], data_shadow['entries']):
        min_nodes = dag.get_minimal_nodes(bn.DAG, nodes, root_nodes)
        diff = min_nodes - datum.keys()

        if any(bn.is_rv(d) for d in diff):
            need_mc = True
        else:
            need_mc = False
        min_nodes = bn.sort(min_nodes)
        leaves.append(DataNodeSet(datum, min_nodes, need_mc))

    return DataBayesianModel(bn, root_nodes, leaves)


def as_simulation_data_model(sm, data, sim_fn, mea_fun, exact_like=False):
    return SimulationBayesianModel(sm, data, sim_fn, mea_fun, exact_like)
