from epidag import *
from scipy.misc import logsumexp
from numpy import log

__author__ = 'TimeWz667'


class BayesianModel:
    DefaultMC = 100

    def __init__(self, dag, group):
        self.DAG = dag
        self.Group = group
        self.InputData = {}
        self.Evidence = [k for k, v in self.Group.items() if v[0] is 'Evidence']
        self.NeedsMC = any([v[0] is 'Mediator' for v in self.Group.values()])

    def input_data(self, dat):
        if list(dat.keys()) < self.Evidence:
            return False

        for e in self.Evidence:
            self.InputData[e] = dat[e]
        return True

    def sample_prior(self):
        vs, prior = dict(), 0
        for k, (ty, loci) in self.Group.items():
            if ty is 'Prior':
                try:
                    dist = loci.get_distribution(vs)
                    v = dist.sample()
                    vs[k] = v
                    prior += dist.logpdf(v)
                except AttributeError:
                    vs[k] = loci.sample(vs)
        return Gene(vs, prior)

    def sample_distribution(self):
        vs, ds = dict(), dict()
        for k, (ty, loci) in self.Group.items():
            if ty is 'Prior':
                try:
                    dist = loci.get_distribution(vs)
                    vs[k] = dist.sample()
                    ds[k] = dist
                except AttributeError:
                    vs[k] = loci.sample(vs)
        return ds

    def evaluate_prior(self, gene):
        return self.DAG.evaluate(gene)

    def evaluate_likelihood(self, gene):
        if not self.InputData:
            return 0

        fixed = dict(gene.Locus)
        fixed.update(self.InputData)
        if self.NeedsMC:
            lis = []
            for _ in range(BayesianModel.DefaultMC):
                li = 0
                mc = self.DAG.sample(fixed)
                for k, (ty, loci) in self.Group.items():
                    if ty is not 'Prior':
                        li += loci.evaluate(mc)
                lis.append(li)
            return logsumexp(lis) - log(BayesianModel.DefaultMC)

        else:
            fixed = self.DAG.sample(fixed)
            li = 0
            for k, (ty, loci) in self.Group.items():
                if ty is not 'Prior':
                    li += loci.evaluate(fixed)
            return li


if __name__ == '__main__':
    pars = '''
    {
        x = 1
        beta0 ~ norm(10, 0.03)
        beta1 ~ norm(0.5, 0.1)
        mu = beta0 + beta1*x
        sigma ~ gamma(0.01, 0.01)
        y ~ norm(mu, sigma)
    }
    '''

    dag1 = BayesianNetwork(pars)

    print(dag1)
    print('Sampling')

    sp1 = dag1.sample()
    print(sp1)

    print('\nPrior probability')
    print(dag1.evaluate(sp1))

    bm1 = dag1.get_bayesian_model(['x', 'y'])

    dat1 = {'x': 1, 'y': 10}

    bm1.input_data(dat1)

    print(bm1.sample_distribution())

    g1 = bm1.sample_prior()
    print(g1)

    print('Likelihood')
    print(bm1.evaluate_likelihood(g1))
