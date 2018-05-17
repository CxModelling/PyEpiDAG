import pandas as pd


__all__ = ['Gene']


class Gene:
    def __init__(self, vs=None, prior=0):
        self.Locus = dict(vs) if vs else dict()
        self.LogPrior = prior
        self.LogLikelihood = 0

    def __len__(self):
        return len(self.Locus)

    def __iter__(self):
        return iter(self.Locus.items())

    def __getitem__(self, item):
        return self.Locus[item]

    def __setitem__(self, key, value):
        self.Locus[key] = value
        self.LogPrior = 0

    def __contains__(self, item):
        return item in self.Locus

    def clone(self):
        g = Gene(self.Locus, self.LogPrior)
        g.LogLikelihood = self.LogLikelihood
        return g

    def __repr__(self):
        if not self.Locus:
            return 'empty'
        loc = [('{}: {:g}' if isinstance(v, float) else '{}: {}').format(k, v) for k, v in self.Locus.items()]
        return ", ".join(loc)

    def to_json(self):
        return {
            'Locus': self.Locus,
            'LogPrior': self.LogPrior,
            'LogLikelihood': self.LogLikelihood
        }

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
