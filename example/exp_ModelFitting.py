import numpy as np
import epidag as dag
__author__ = 'TimeWz667'


class Agent:
    def __init__(self, name, p):
        self.Name = name
        self.Parameters = p
        self.X = p.get_sampler('x')

    def produce(self, k):
        return self.X.sample(k)


class AgentBasedModel:
    def __init__(self, pars, n_agents):
        self.Parameters = pars
        self.Agents = list()
        for i in range(n_agents):
            name = 'Ag{}'.format(i)
            p = pars.breed(name, 'ag')
            self.Agents.append(Agent(name, p))

    def product(self, k):
        return np.array([ag.produce(k) for ag in self.Agents]).sum()


def fn_sim(pars, data):
    abm = AgentBasedModel(pars, data['N'])
    return abm.product(data['K'])


def fn_mean(sim, data):
    return -abs(sim - data['X'])


d = {
    'N': 10,
    'K': 10,
    'X': 200
}

script = '''
PCore BetaBin {
    al = 1
    be = 1
    p ~ beta(al, be)
    x ~ binom(5, p)
}
'''

bn = dag.bayes_net_from_script(script)

ns = dag.NodeSet('root', as_fixed=['p'])
ns.new_child('ag', as_floating=['x'])

sm = dag.as_simulation_core(bn, ns)
sm.deep_print()

sdm = dag.as_simulation_data_model(sm, d, fn_sim, fn_mean)

fit = dag.fitting.GA(sdm)
fit.fit(100)
print(fit.BestFit)
