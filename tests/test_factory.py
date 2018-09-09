import unittest
import epidag as dag
import epidag.factory as fac
import epidag.bayesnet.distribution as dist


class TestParseFunction(unittest.TestCase):
    def test_not_valid_function(self):
        with self.assertRaises(SyntaxError):
            dag.parse_function('exp[rate=0.4]')

    def test_one_argument(self):
        fn = dag.parse_function('exp(0.4)')
        self.assertEqual(fn.Function, 'exp')
        self.assertEqual(fn.get_arguments()[0]['value'], 0.4)

    def test_many_arguments(self):
        fn = dag.parse_function('gamma(0.4, 1/1000)')
        args = [arg['value'] for arg in fn.get_arguments()]
        self.assertListEqual(args, [0.4, 0.001])

    def test_key_argument(self):
        fn = dag.parse_function('gamma(0.4, scale=1/1000)')
        self.assertEqual(fn.get_arguments()[1]['key'], 'scale')

    def test_dict_argument(self):
        fn = dag.parse_function('cat(kv={"a": 4, "b": 3})')
        self.assertDictEqual({'a': 4, 'b': 3}, fn.get_arguments()[0]['value'])

    def test_list_argument(self):
        fn = dag.parse_function('list(vs=[2, "hh", 4])')
        self.assertListEqual(fn.get_arguments()[0]['value'], [2, "hh", 4])

    def test_two_step(self):
        fn = dag.parse_function('gamma(shape=0.01, rate=0.02)')
        self.assertEqual(fn.Function, 'gamma')
        arg_keys = [arg['key'] for arg in fn.get_arguments()]
        self.assertListEqual(arg_keys, ['shape', 'rate'])
        di = dist.parse_distribution(fn)
        self.assertEqual(di.mean(), 0.5)


class TestArguments(unittest.TestCase):
    def try_truthy(self, arg, truthy, res=None):
        self.assertTrue(arg(truthy, res))

    def try_falsy(self, arg, falsy, res=None):
        with self.assertRaises(fac.ValidationError):
            arg(falsy, res)

    def test_positive_float(self):
        arg = fac.PositiveFloat('+1.0')
        self.try_truthy(arg, 1)
        self.try_falsy(arg, -1)

    def test_negative_integer(self):
        arg = fac.NegativeInteger('-1')
        self.try_truthy(arg, -1)
        self.try_falsy(arg, 1)

    def test_regex(self):
        arg = fac.RegExp('A-B', r'\w+-\w+')
        self.try_truthy(arg, 'A-B')
        self.try_falsy(arg, 'A+B')

    def test_options(self):
        arg = fac.Options('A, B, C', ['A', 'B', 'C'])
        self.try_truthy(arg, 'A')
        self.try_falsy(arg, 'D')

    def test_options_with_resource(self):
        arg = fac.Options('A, B, C', 'ABC')
        self.try_truthy(arg, 'A', res={'ABC': ['A', 'B', 'C']})
        self.try_falsy(arg, 'D', res={'ABC': ['A', 'B', 'C']})

    def test_options_with_resource_objects(self):
        from collections import namedtuple
        entry = namedtuple('entry', 'Name')

        arg = fac.Options('A, B, C', 'ABC')
        self.try_truthy(arg, 'A', res={'ABC': {'A': entry('A'), 'B': entry('B'), 'C': entry('C')}})
        self.try_falsy(arg, 'D', res={'ABC': {'A': entry('A'), 'B': entry('B'), 'C': entry('C')}})

    def test_prob_tab(self):
        arg = fac.ProbTab('k,v')
        res = {'ABC': {'A': 0.2, 'B': 0.3, 'C': 0.5},
               'BC': {'A': 'A', 'B': 0.3, 'C': 0.5}}
        self.try_truthy(arg, 'ABC', res=res)
        self.try_falsy(arg, 'D', res=res)
        self.try_falsy(arg, 'BC', res=res)


if __name__ == '__main__':
    unittest.main()
