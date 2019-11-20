import unittest
import logging
import sys
import epidag as dag

scr = '''
PCore test {
    al = 1
    be = 1
    prob ~ beta(al, be)
    x ~ binom(n, prob)    
}
'''


data = [
    {'id': 1, 'n': 10, 'x': 4},
    {'id': 2, 'n': 20, 'x': 7}
]


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class FittingTest(unittest.TestCase):
    def setUp(self):
        self.BN = dag.bayes_net_from_script(scr)
        self.DM = dag.dag.as_data_model(self.BN, data)
        self.Logger = logging.getLogger(__name__)

    def test_sir(self):
        print()
        fit = dag.fitting.SIR(self.DM)
        fit.log_on(self.Logger)
        fit.fit(1000)
        print(fit.summarise_posterior())

    def test_abc(self):
        print()
        fit = dag.fitting.ABC(self.DM)
        fit.log_on(self.Logger)
        fit.fit(100)
        print(fit.summarise_posterior())

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
