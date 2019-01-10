import unittest
import epidag as dag

script_betabin = '''
PCore BetaBi {
    al = 1
    be = 1
    p ~ beta(al, be)
    x ~ binom(n, p)
}
'''


script_betabin2 = '''
PCore BetaBi {
    al = 1
    be = 1
    p ~ beta(al, be)
    x1 ~ binom(2, p)
    x2 ~ binom(10, p)
}
'''


class ParameterCoreCloneTest(unittest.TestCase):
    def test_simple(self):
        bn = dag.bayes_net_from_script(script_betabin)
        sc = dag.as_simulation_core(bn)
        pc = sc.generate("T1", {'n': 10})
        self.assertEqual(pc['n'], 10)

    def test_random(self):
        bn = dag.bayes_net_from_script(script_betabin)
        sc = dag.as_simulation_core(bn, random=['p'])
        pc = sc.generate("T2", {'n': 10})
        self.assertListEqual(list(pc.get_samplers().keys()), ['x'])

    def test_no_exo(self):
        bn = dag.bayes_net_from_script(script_betabin)
        sc = dag.as_simulation_core(bn)
        pc = sc.generate("T3")
        with self.assertRaises(KeyError):
            pc.get_sampler('x')()

    def test_div(self):
        bn = dag.bayes_net_from_script(script_betabin2)

        hei = {
            'root': ['b', 'a'],
            'a': ['x1'],
            'b': ['x2']
        }

        sc = dag.as_simulation_core(bn, hei)
        pc = sc.generate("T4")
        pc_a = pc.breed('A', 'a')
        pc_b = pc.breed('B', 'b')

        self.assertIn('x1', pc_a.get_samplers())
        self.assertIn('x2', pc_b.get_samplers())

    def test_conflict(self):
        bn = dag.bayes_net_from_script(script_betabin2)

        hei = {
            'root': ['b', 'a'],
            'a': ['x1', 'p'],
            'b': ['x2']
        }

        sc = dag.as_simulation_core(bn, hei)
        pc = sc.generate("T5")
        pc_a = pc.breed('A', 'a')
        pc_b = pc.breed('B', 'b')

        with self.assertRaises(KeyError):
            pc_a.get_sampler('p')

        with self.assertRaises(KeyError):
            pc_b.get_sampler('p')

        self.assertIn('p', pc.get_samplers())


if __name__ == '__main__':
    unittest.main()
