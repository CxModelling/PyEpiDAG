import unittest
import epidag as dag


class ParameterCoreCloneTest(unittest.TestCase):
    def test_clone(self):
        script = '''
        PCore Regression {
        x = 1
        y = x + 1
        z = y + 1
        }
        '''

        bn = dag.bayes_net_from_script(script)

        hei = {
            'a': ['b', 'x'],
            'b': ['c', 'y'],
            'c': ['z']
        }

        sc = dag.as_simulation_core(bn, hei)

        pc_a = sc.generate('A')
        pc_b = pc_a.breed('B', 'b')
        pc_c = pc_b.breed('C', 'c')

        pc_aa = pc_a.clone(copy_sc=True)
        pc_cc = pc_aa.find_descendant('A@B@C')
        self.assertEqual(pc_c['z'], 3)
        self.assertEqual(pc_cc['z'], 3)

        pc_aa.impulse({'x': 5})
        self.assertEqual(pc_c['z'], 3)
        self.assertEqual(pc_cc['z'], 7)

        pc_a.impulse({'x': 7})
        self.assertEqual(pc_c['z'], 9)
        self.assertEqual(pc_cc['z'], 7)


if __name__ == '__main__':
    unittest.main()
