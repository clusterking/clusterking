#!/usr/bin/python3

import unittest

from modules.distribution import *


class TestDistribution(unittest.TestCase):
    def test_bin_function(self):
        self.assertEqual(bin_function(lambda x: 1, [1, 2, 3]),
                         [1, 1])
        self.assertEqual(bin_function(lambda x: 1, [1, 2, 3], normalized=True),
                         [0.5, 0.5])
        self.assertEqual(bin_function(lambda x: x, [0, 1]),
                         [0.5])
        self.assertEqual(bin_function(lambda x: x, [0, 1], midpoints=True),
                         [(0.5, 0.5)])
        self.assertEqual(bin_function(lambda x: x, [0, 1], midpoints=True,
                                      normalized=True),
                         [(0.5, 1)])
        # bin_function(lambda x: x**2, [1, ])
