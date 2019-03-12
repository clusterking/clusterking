#!/usr/bin/python3

# std
import unittest

# 3rd party
import numpy as np

# ours
from physics.models.bdlnu import bin_function


class TestDistribution(unittest.TestCase):
    def test_bin_function(self):
        self.assertSequenceEqual(
            list(bin_function(lambda x: 1, np.array([1, 2, 3]))),
            [1, 1])
        self.assertSequenceEqual(
            list(bin_function(lambda x: 1, np.array([1, 2, 3]),
                              normalized=True)),
            [0.5, 0.5])
        self.assertSequenceEqual(
            list(bin_function(lambda x: x, np.array([0, 1]))),
            [0.5])
