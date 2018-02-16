import pandas as pd


class Gene:
    def __init__(self, vs=None, prior=0):
        self.Locus = {k: v for k, v in vs.items()} if vs else dict()
        self.LogPrior = prior
        self.LogLikelihood = 0

    def __len__(self):
        return len(self.Locus)

    def __getitem__(self, item):
        return self.Locus[item]

    def __setitem__(self, key, value):
        self.Locus[key] = value
        self.LogPrior = 0

    def clone(self):
        g = Gene(self.Locus, self.LogPrior)
        g.LogLikelihood = self.LogLikelihood
        return g

    def __repr__(self):
        return ", ".join(['{}: {:g}'.format(k, v) for k, v in self.Locus.items()])

    @property
    def LogPosterior(self):
        return self.LogPrior + self.LogLikelihood

    @staticmethod
    def summarise(genes):
        df = pd.DataFrame([gene.Locus for gene in genes])
        return df.describe()

    @staticmethod
    def mean(genes):
        df = pd.DataFrame([gene.Locus for gene in genes])
        return dict(df.mean())

    @staticmethod
    def to_json(file, genes):
        df = pd.DataFrame([gene.Locus for gene in genes])
        df.to_json(file, orient='records')

    @staticmethod
    def to_csv(file, genes):
        df = pd.DataFrame([gene.Locus for gene in genes])
        df.to_csv(file)
