import unittest
from epidag.bayesnet.loci import DistributionLoci, FunctionLoci
from epidag.simulation.actor import *


class ActorTest(unittest.TestCase):
    def test_frozen_single(self):
        f1 = FrozenSingleActor('A', DistributionLoci('A', 'k(a)'), ['a'])
        f1.update({'a': 1})
        self.assertEqual(1.0, f1.sample())

        f1.update({'a': 3})
        self.assertEqual(3.0, f1.sample())

        f2 = FrozenSingleActor('B', FunctionLoci('B', 'b+4'), ['b'])
        f2.update({'b': 1})
        self.assertEqual(5.0, f2.sample())

        f2.update({'b': 3})
        self.assertEqual(7.0, f2.sample())

        self.assertEqual(7.0, f2.Sampler)

    def test_single(self):
        s1 = SingleActor('C', DistributionLoci('C', 'k(c)'), ['c'])
        self.assertRaises(KeyError, s1.sample)
        self.assertEqual(3, s1.sample({'c': 3}))

    def test_compound(self):
        c1 = CompoundActor('D', DistributionLoci('D', 'k(d)'),
                           ['b'],
                           [
                               DistributionLoci('a', 'k(1)'),
                               FunctionLoci('c', 'a+b'),
                               FunctionLoci('d', 'c+1')
                           ])
        self.assertEqual('(b)->(a,c,d)->k(d)', str(c1))
        self.assertEqual(6, c1.sample({'b': 4}))

        c2 = CompoundActor('D', FunctionLoci('D', 'pow(d, 2)'),
                           ['b'],
                           [
                               DistributionLoci('a', 'k(1)'),
                               FunctionLoci('c', 'a+b'),
                               FunctionLoci('d', 'c+1')
                           ])

        self.assertEqual('(b)->(a,c,d)->pow(d, 2)', str(c2))
        self.assertEqual(36, c2.sample({'b': 4}))

    def test_sampler(self):
        sam1 = Sampler(SingleActor('E', DistributionLoci('E', 'k(e)'), ['e']), {'e': 4})
        self.assertEqual(4, sam1.sample())


if __name__ == '__main__':
    unittest.main()
