import pandas as pd
from epidag.bayesnet.distribution import CategoricalRV
from .frame import AbsDataSet

__author__ = 'TimeWz667'
__all__ = ['TransitionMatrix', 'ConditionalProbabilityTable']


class TransitionMatrix(AbsDataSet):
    def __init__(self, mat):
        AbsDataSet.__init__(self, mat)
        self.Transitions = {k: CategoricalRV(dict(ir)) for k, ir in mat.iterrows()}
        self.StateFrom = list(mat.index)
        self.StateTo = list(mat.columns)

    def __call__(self, current):
        return self.sample_next(current)

    def __getitem__(self, k):
        try:
            return self.Transitions[k]
        except KeyError:
            raise KeyError('Unknown state: {}'.format(k))

    def sample_next(self, current):
        return self[current]()

    def sample(self, n=1, **kwargs):
        if n == 1:
            return self(current=kwargs)
        else:
            if 'currents' in kwargs:
                ks = kwargs['currents']
            else:
                ks = [kwargs['current']]
            n = max(n, len(ks))
            return [self(current=k) for k in zip(range(n), ks)]

    def __repr__(self):
        return 'Transition Matrix: {}'.format(', '.join(self.StateFrom))


class ConditionalProbabilityTable(AbsDataSet):
    def __init__(self, mat, indices, i_prob):
        AbsDataSet.__init__(self, mat)
        self.Indices = indices
        self.IndexProb = i_prob
        self.Ps = mat[i_prob]
        self.Ps /= sum(self.Ps)
        self.Ps = list(self.Ps)
        self.Groups = dict()
        self.InvGroups = dict()

        for i, (k, ir) in enumerate(mat.iterrows()):
            key = ':'.join([ir[x] for x in indices])
            p = self.Ps[i]
            self.InvGroups[key] = p
            self.Groups[key] = {x: ir[x] for x in indices}
        self.Sampler = CategoricalRV(self.InvGroups)

    def __call__(self):
        return self.Groups[self.Sampler()]

    def marginalise(self, index):
        ix_new = [x for x in self.Indices if x != index]
        df = self.SourceMatrix.groupby(ix_new)[self.IndexProb].sum()
        df = pd.DataFrame(df).reset_index()
        return ConditionalProbabilityTable(df, ix_new, self.IndexProb)

    def condition(self, **kwargs):
        ix_cond = [x for x in self.Indices if x in kwargs]
        mat = self.SourceMatrix

        for k in ix_cond:
            mat = mat[mat[k] == kwargs[k]]

        ix_new = [x for x in self.Indices if x not in ix_cond]
        mat = mat.drop(ix_cond, axis=1)
        return ConditionalProbabilityTable(mat, ix_new, self.IndexProb)

    def __repr__(self):
        return 'ConditionalProbabilityTable: {}| {}'.format(self.IndexProb, ', '.join(self.Indices))
