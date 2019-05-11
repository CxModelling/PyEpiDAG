from epidag.bayesnet import parse_distribution
from collections import Counter

__author__ = 'TimeWz667'


dists = [
        'exp(0.01)',
        'gamma(0.01, 1)',
        'lnorm(0.5, 1)',
        'k(1)',
        'unif(0, 1)',
        'chisq(20)',
        'triangle(2, 3, 5)',
        'binom(size=4, prob=0.5)'
    ]

for di in dists:
    print('p(x): \t', di)
    di = parse_distribution(di)
    print('JSON:\t', di.to_json())
    print('Mean:\t', di.mean())
    print('Std:\t', di.std())
    print('')


di = 'cat(kv={"M": 3411,"O": 3502,"Y": 52})'
print('p(x): \t', di)
dist_cat = parse_distribution(di)

print('JSON:\t', dist_cat.to_json())
print('Counts:\t', Counter(dist_cat.render(10000)))
