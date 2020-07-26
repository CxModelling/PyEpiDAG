from epidag.bayesnet import Chromosome

__author__ = 'TimeWz667'
__all__ = ['Result', 'BayesResult', 'FrequentistResult']


class Result:
    def __init__(self, nodes, model):
        self.Benchmarks = dict()
        self.Nodes = nodes
        self.Model = model

    def __setitem__(self, key, value):
        self.Benchmarks[key] = value

    def to_json(self):
        return {
            'Name': self.Model.Name,
            'Pars': [node.to_json() for node in self.Nodes],
            'Benchmarks': dict(self.Benchmarks)
        }

    def to_df(self):
        return Chromosome.to_data_frame(self.Nodes)

    def summarise(self):
        return self.to_df().describe()

    def to_csv(self, file):
        df = self.to_df()
        df.to_csv(file)


class BayesResult(Result):
    def __init__(self, nodes, model, alg):
        Result.__init__(self, nodes, model)
        self.Benchmarks['ESS'] = 0
        self.Benchmarks['Rhat'] = []
        self.Benchmarks['Niter'] = 0
        self.Fitter = alg

    def MAP(self):
        return min(self.Nodes)

    def to_json(self):
        js = Result.to_json(self)
        js['MAP'] = self.MAP()
        return js


class FrequentistResult(Result):
    def __init__(self, node, model):
        Result.__init__(self, [node], model)
