import unittest
from epidag.monitor import Monitor

__author__ = 'TimeWz667'


class TestMonitor(unittest.TestCase):
    def setUp(self):
        self.Mon = Monitor('Test')

    def test_recording(self):
        self.Mon.keep(a=1)
        self.assertEqual(self.Mon['a'], 1)
        self.Mon.step()
        self.assertEqual(len(self.Mon.Records), 1)
        self.assertEqual(self.Mon.Time, 1)

        with self.assertRaises(KeyError):
            self.Mon.step(0.5)

    def test_reset(self):
        self.Mon.reset(50)
        self.assertEqual(self.Mon.Time, 50)
        self.Mon.keep(a=1)
        self.assertEqual(self.Mon['a'], 1)
        self.Mon.step(53)
        self.assertEqual(len(self.Mon.Records), 1)
        self.assertEqual(self.Mon.Time, 53)


if __name__ == '__main__':
    unittest.main()
