import unittest

import epidag.factory as fac


class TestParseFunction(unittest.TestCase):
    def test_not_valid_function(self):
        with self.assertRaises(ValueError):
            fac.parse_function('exp[0.4]')

    def test_one_argument(self):
        func, args = fac.parse_function('exp(0.4)')
        self.assertEqual(func, 'exp')
        self.assertIn(0.4, args)

    def test_many_arguments(self):
        _, args = fac.parse_function('gamma(0.4, 1/1000)')
        self.assertListEqual([0.4, 0.001], args)

    def test_key_argument(self):
        _, args = fac.parse_function('gamma(0.4, v=1/1000)')
        self.assertListEqual([0.4, ('v', 0.001)], args)

    def test_dict_argument(self):
        _, args = fac.parse_function('cat(kv={"a": 4, "b": 3})')
        self.assertDictEqual({'a': 4, 'b': 3}, args[0][1])

    def test_list_argument(self):
        _, args = fac.parse_function('list(vs=[2, "hh", 4])')
        self.assertListEqual([2, "hh", 4], args[0][1])


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
