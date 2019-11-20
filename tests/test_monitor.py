import unittest

import epidag.factory as fac


__author__ = 'TimeWz667'

class TestParseFunction(unittest.TestCase):
    def test_not_valid_function(self):
        with self.assertRaises(SyntaxError):
            dag.parse_function('exp[rate=0.4]')

    def test_one_argument(self):
        fn = dag.parse_function('exp(0.4)')
        self.assertEqual(fn.Function, 'exp')
        self.assertEqual(fn.get_arguments()[0]['value'], 0.4)

