import unittest
from epidag.util import *


class ExpressionParserCase(unittest.TestCase):

    def setUp(self):
        self.me = parse_math_expression('x+y/2 * max(z, 5)')

    def test_parents(self):
        self.assertSetEqual(self.me.Var, {'x', 'y', 'z'})
        self.assertSetEqual(self.me.Func, {'max'})

    def test_call(self):
        self.assertEqual(self.me(), 'x+y/2 * max(z, 5)')
        self.assertEqual(self.me({'x': 2, 'y': 4, 'z': 10}), 22)


class FunctionParserCase(unittest.TestCase):
    def setUp(self):
        self.fn = parse_function('my_func(4*a, "k", k, t=5, s=False)')

    def test_definition(self):
        self.assertEqual(str(self.fn), 'my_func(4*a,"k",k,t=5,s=False)')
        self.assertEqual(self.fn.Function, 'my_func')

    def test_json(self):
        self.assertListEqual(self.fn.to_json()['Args'], ['(4 * a)', 'k', 'k', 5, False])
        self.assertListEqual(self.fn.to_json({'k': 7})['Args'], ['(4 * a)', 'k', 7, 5, False])
        self.assertListEqual(self.fn.to_json({'k': 7, 'a': 10})['Args'], [40, 'k', 7, 5, False])

    def test_order(self):
        self.assertListEqual(self.fn.to_json()['Args'], ['(4 * a)', 'k', 'k', 5, False])


if __name__ == '__main__':
    unittest.main()
