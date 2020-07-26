from abc import ABCMeta, abstractmethod
import scipy.stats as stats
from epidag.bayesnet import Chromosome
from epidag.fn import evaluate_nodes, sample_minimally

__author__ = 'TimeWz667'
__all__ = ['BayesianModel']


class BayesianModel(metaclass=ABCMeta):
    def __init__(self, bn, pars):
        self.BN = bn
        self.Name = bn.Name
        self.ParameterNodes = [p for p in pars if self.BN.is_rv(p)]

    def sample_prior(self):
        ps, src = sample_minimally(self.BN, self.ParameterNodes)
        src.update(ps)
        return Chromosome(src)

    def evaluate_prior(self, prior):
        prior.LogPrior = evaluate_nodes(self.BN, prior)
        return prior.LogPrior

    @property
    def MovableNodes(self):
        p = self.sample_prior()

        res = []
        for node in self.ParameterNodes:
            d = self.BN[node].get_distribution(p)
            res.append({'Name': node, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
        return res

    @property
    def has_exact_likelihood(self):
        return False

    @abstractmethod
    def evaluate_distance(self, pars):
        pass

    def evaluate_likelihood(self, pars):
        return stats.norm.pdf(stats.sqrt(self.evaluate_distance(pars)))


if __name__ == '__main__':
    from epidag import bayes_net_from_script

    class BinBeta(BayesianModel):
        def __init__(self, bn, data):
            BayesianModel.__init__(self, bn, pars=['prob'])
            self.Data = data

        @property
        def has_exact_likelihood(self):
            return True

        def evaluate_distance(self, pars):
            di = 0
            for d in self.Data:
                pars.update(d)
                di += self.BN['x'].evaluate(pars)
            return di

        def evaluate_likelihood(self, pars):
            return - self.evaluate_distance(pars)


    data = [
        {'id': 1, 'n': 10, 'x': 4},
        {'id': 2, 'n': 20, 'x': 7}
    ]

    bn = bayes_net_from_script('''
    PCore test {
        prob ~ beta(1, 1)
        x ~ binom(n, prob)    
    }
    ''')

    model = BinBeta(bn, data)

    p = model.sample_prior()
    print(p)
