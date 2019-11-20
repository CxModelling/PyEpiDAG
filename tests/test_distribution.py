import unittest
from collections import Counter
import numpy as np
from epidag.distribution import *

__author__ = 'TimeWz667'


class DistributionCase(unittest.TestCase):
    def test_uniform(self):
        d = parse_distribution('unif(10, 15)')
        self.assertEqual(d.mean(), 12.5)
        self.assertAlmostEqual(pow(d.std(), 2), 25/12)
        self.assertEqual(d.Upper, 15)
        self.assertEqual(d.Lower, 10)

    def test_exponential(self):
        d = parse_distribution('exp(10)')
        self.assertEqual(d.mean(), 0.1)
        self.assertAlmostEqual(d.std(), 0.1)
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, 0)

    def test_normal(self):
        d = parse_distribution('norm(10, 1)')
        self.assertEqual(d.mean(), 10)
        self.assertAlmostEqual(d.std(), 1)
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, float('-inf'))

    def test_gamma(self):
        d = parse_distribution('gamma(0.1, 0.1)')
        self.assertEqual(d.mean(), 1)
        self.assertAlmostEqual(pow(d.std(), 2), 10)
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, 0)

    def test_invgamma(self):
        d = parse_distribution('invgamma(5, 1)')
        self.assertEqual(d.mean(), 1/4)
        self.assertEqual(pow(d.std(), 2), 1/16/3)
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, 0)

    def test_beta(self):
        d = parse_distribution('beta(1, 0.5)')
        self.assertEqual(d.mean(), 2/3)
        self.assertAlmostEqual(d.std(), np.sqrt(0.5/2.25/2.5))
        self.assertEqual(d.Upper, 1)
        self.assertEqual(d.Lower, 0)

    def test_lnorm(self):
        d = parse_distribution('lnorm(0.5, 1)')
        mu = np.exp(0.5)
        std = np.exp(1)
        self.assertAlmostEqual(d.mean(), np.exp(mu + pow(std, 2)/2))
        self.assertAlmostEqual(d.std(), np.sqrt(np.exp(2*mu + std*std) * (np.exp(std*std) - 1)))
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, 0)

    def test_binom(self):
        d = parse_distribution('binom(5, 0.5)')
        self.assertEqual(d.mean(), 2.5)
        self.assertEqual(d.std(), np.sqrt(5) * 0.5)
        self.assertEqual(d.Upper, 5)
        self.assertEqual(d.Lower, 0)

    def test_pois(self):
        d = parse_distribution('pois(5)')
        self.assertEqual(d.mean(), 5)
        self.assertEqual(d.std(), np.sqrt(5))
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, 0)

    def test_chisq(self):
        d = parse_distribution('chisq(10)')
        self.assertEqual(d.mean(), 10)
        self.assertAlmostEqual(pow(d.std(), 2), 20)
        self.assertEqual(d.Upper, float('inf'))
        self.assertEqual(d.Lower, 0)

    def test_triangle(self):
        d = parse_distribution('triangle(10, 15, 20)')
        self.assertEqual(d.mean(), 15)
        self.assertAlmostEqual(pow(d.std(), 2), (725-650) / 18)
        self.assertEqual(d.Upper, 20)
        self.assertEqual(d.Lower, 10)

    def test_const(self):
        d = parse_distribution('k(10)')
        self.assertEqual(d.mean(), 10)
        self.assertEqual(d.std(), 0)
        self.assertEqual(d.Upper, 10)
        self.assertEqual(d.Lower, 10)


class CategoricalRVCase(unittest.TestCase):
    def test_def(self):
        cat = parse_distribution('cat({"Y": 50, "M": 450,"O": 500})')
        js = cat.to_json()
        self.assertDictEqual(js['Args']['kv'], {"Y": 50, "M": 450, "O": 500})
        self.assertListEqual(list(cat.p), [0.05, 0.45, 0.5])

    def test_sample(self):
        cat = parse_distribution('cat({"Y": 50, "M": 450,"O": 500})')
        res = Counter(cat.sample(10000))
        self.assertSetEqual(set(res.keys()), set(cat.cat))
        self.assertGreater(res['O'], 4000)


if __name__ == '__main__':
    unittest.main()
