import unittest
import logging
import sys
import epidag as dag

scr = '''
PCore test {
    prob ~ beta(1, 1)
    x ~ binom(n, prob)    
}
'''

class BinBeta(dag.fitting.BayesianModel):
    def __init__(self, bn, data):
        dag.fitting.BayesianModel.__init__(self, bn, pars=['prob'])
        self.Data = data

    @property
    def has_exact_likelihood(self):
        return True

    def evaluate_distance(self, pars):
        return - self.evaluate_likelihood(pars)

    def evaluate_likelihood(self, pars):
        li = 0
        pars = dict(pars)
        for d in self.Data:
            pars.update(d)
            li += self.BN['x'].evaluate(pars)
        return li

data = [
    {'id': 1, 'n': 10, 'x': 4},
    {'id': 2, 'n': 20, 'x': 7}
]


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class FittingTest(unittest.TestCase):
    def setUp(self):
        self.BN = dag.bayes_net_from_script(scr)
        self.DM = BinBeta(self.BN, data)
        self.Logger = logging.getLogger(__name__)

    def test_sir(self):
        print()
        alg = dag.fitting.SIR()
        res = alg.fit(self.DM, n_post=1000)

        print(res.summarise())
        print(res.Benchmarks)


    def test_abc(self):
        print()
        alg = dag.fitting.ABC()
        res = alg.fit(self.DM, n_post=1000)

        print(res.summarise())
        print(res.Benchmarks)

    def test_mcmc(self):
        print()
        fit = dag.fitting.MCMC(self.DM)
        fit.fit(1000)
        print(fit.summarise_posterior())

    def test_ga(self):
        print()
        fit = dag.fitting.GA(self.DM)
        fit.fit(500, target='MAP')
        print(fit.summarise_fitness())


#if __name__ == '__main__':
#    unittest.main()
