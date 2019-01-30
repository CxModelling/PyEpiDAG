import pandas as pd
import networkx as nx

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

    def keys(self):
        return self.Locus.keys()

    def impulse(self, new_locus, bn=None):
        """
        Change the value of some locus
        :param new_locus: {name: value}; value = None if prior applied
        :type new_locus: dict
        :param bn: source bayesian network; None if no check needed
        :type bn: BayesNet
        :return:
        """
        if bn:
            g = bn.DAG
            imp = {k: v for k, v in new_locus.items() if k in self}
            shocked = set.union(*[set(nx.descendants(g, k)) for k in imp.keys()])
            non_imp = [k for k, v in imp.items() if v is None]
            imp = {k: v for k, v in imp.items() if v is not None}
            shocked.difference_update(imp.keys())
            shocked = shocked.union(non_imp)
            shocked.intersection_update(self.Locus.keys())
            self.Locus.update(imp)

            for nod in bn.Order:
                if nod in shocked:
                    self[nod] = g.nodes[nod]['loci'].sample(self)

            if imp:
                self.LogLikelihood = 0

        else:
            imp = {k: v for k, v in new_locus.items() if k in self}
            self.Locus.update(imp)
            if imp:
                self.LogLikelihood = 0

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
