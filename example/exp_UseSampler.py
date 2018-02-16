from collections import Counter
from epidag.bayesnet import Sampler, resample
from numpy import log

__author__ = 'TimeWz667'

print('Use sampler')
sam = Sampler([1, 2, 3, 4])
ct = Counter(sam(10000))
print(ct)

print('Resampling')
x, y, p = resample(log([0.2, 0.05, 0.795]), ['A', 'B', 'C'], ['a', 'b', 'c'])
print(x)
print(y)
print(p)






