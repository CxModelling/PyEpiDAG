import pandas as pd
import networkx as nx

__all__ = ['Chromosome']


class Chromosome:
    def __init__(self, vs=None, prior=None):
        self.Locus = dict(vs) if vs else dict()
        self.LogPrior = prior
        self.LogLikelihood = None

    def __len__(self):
        return len(self.Locus)

    def __iter__(self):
        return iter(self.Locus.items())

    def __bool__(self):
        return True

    def __getitem__(self, item):
        return self.Locus[item]

    def __setitem__(self, key, value):
        self.Locus[key] = value
        self.reset_probability()

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
                    self[nod] = g.nodes[nod]['loci'].render(self)

            if imp:
                self.reset_probability()

        else:
            imp = {k: v for k, v in new_locus.items() if k in self}
            self.Locus.update(imp)
            if imp:
                self.reset_probability()

    def clone(self):
        g = Chromosome(self.Locus, self.LogPrior)
        g.LogLikelihood = self.LogLikelihood
        return g

    def reset_probability(self):
        self.LogLikelihood = None
        self.LogPrior = None

    def is_prior_evaluated(self):
        return self.LogPrior is not None

    def is_likelihood_evaluated(self):
        return self.LogLikelihood is not None

    def is_evaluated(self):
        return self.is_prior_evaluated() and self.is_likelihood_evaluated()

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

    def to_data(self):
        vs = dict(self.Locus)
        if self.is_prior_evaluated():
            vs['LogPrior'] = self.LogPrior
        if self.is_likelihood_evaluated():
            vs['LogLikelihood'] = self.LogLikelihood
        if self.is_evaluated():
            vs['LogPosterior'] = self.LogPosterior

        return vs

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
    def to_data_frame(genes):
        return pd.DataFrame([gene.to_data() for gene in genes])
