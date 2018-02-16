from epidag.bayesnet import Gene
import pandas as pd
import numpy as np
import numpy.random as rd


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
