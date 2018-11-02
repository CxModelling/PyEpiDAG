import pandas as pd
from epidag.data import *

__author__ = 'TimeWz667'


# Transition Matrix
print('\nTransition Matrix')
mat = pd.read_csv('../data/transition.csv', index_col=0)
trm = TransitionMatrix(mat)
print(trm)
print('\nSampling\n')
st = 'A'
print('State', st)
for _ in range(5):
    st = trm.sample_next(st)
    print('State', st)

print(trm)


print('\nConditional Probability Table')
mat = pd.read_csv('../data/condprobtab.csv')

cpt = ConditionalProbabilityTable(mat, ['Age', 'Sex'], 'Prob')


print(cpt)

print('\nMarginalisation\n', cpt.marginalise('Age'))

print('\nConditioning\n', cpt.condition(Age='O'))

print('\nSampling\n', cpt())


print('\nTime Series')
mat = pd.read_csv('../data/ts.csv')

ts = TimeSeries(mat, 'Time', 'X', 'linear')

print(ts)
print('\nSampling\n', ts(range(10)))


print('\nTime Series Vector')
mat = pd.read_csv('../data/ts.csv')

tsv = TimeSeriesVector(mat, 'Time', ['X', 'Y'], 'linear')

print(tsv)
print('\nSampling\n', pd.DataFrame.from_dict(tsv([1, 2, 3])))


print('\nTime Series Probability Table')
mat = pd.read_csv('../data/tspt.csv')

tspt = TimeSeriesProbabilityTable(mat, 'Time', ['X', 'Y', 'Z'])

print(tspt)
print('\nSampling\n', tspt.sample(n=10, t=5))


print('\nLee Carter')
mat_a = pd.read_csv('../data/lca.csv')
mat_t = pd.read_csv('../data/lct.csv')


lc = LeeCarter(mat_t=mat_t, mat_a=mat_a, i_time='Time', i_age='Age',
               i_al='Alpha', i_be='Beta', i_ka='Kappa')


print('\nGet single rate\n', lc.get_rate(5, 'M'))

print('\nGet Rate Table\n', lc(5))
