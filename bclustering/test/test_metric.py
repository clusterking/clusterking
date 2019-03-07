#!/usr/bin/env python3

# std
import unittest

# 3rd
import numpy as np

# ours
from bclustering.metric import *


class MyTestCase(unittest.TestCase):
    """ Implements an additional general testing methods. """
    def assertAllClose(self, a, b):
        """ Compares two numpy arrays """
        self.assertTrue(np.allclose(
            a, b
        ))


class TestMetricUtils(MyTestCase):
    """ To test the conversion functions cov2err etc.
    Subclassed for specific test cases """
    def setUp(self):
        self.cov = None
        self.corr = None
        self.err = None

    def test_cov2err(self):
        if self.cov is None:
            return
        self.assertAllClose(
            cov2err(self.cov),
            self.err
        )

    def test_corr2cov(self):
        if self.cov is None:
            return
        self.assertAllClose(
            corr2cov(self.corr, self.err),
            self.cov
        )

    def test_cov2corr(self):
        if self.cov is None:
            return
        self.assertAllClose(
            cov2corr(self.cov),
            self.corr
        )


class TestMetricUtils1D(TestMetricUtils):
    """ Test metric utils with 1 datapoint.
    Testing methods inherited from TestMetricUtils
    """
    def setUp(self):
        self.cov = np.array([[4, 4], [-4, 16]])
        self.err = np.array([2, 4])
        self.corr = np.array([[1, 1/2], [-1/2, 1]])


class TestMetricUtils2D(TestMetricUtils):
    """ Test metric utils with several datapoints
    Testing methods inherited from TestMetricUtils
    """
    def setUp(self):
        self.cov = np.array([
            [[4, 4], [-4, 16]],
            [[4, 0], [0, 25]]
        ])
        self.err = np.array([
            [2, 4],
            [2, 5]
        ])
        self.corr = np.array([
            [[1, 1/2], [-1/2, 1]],
            [[1, 0], [0, 1]]
        ])


class TestDataWithErrors(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [100, 200],
            [400, 500]
        ])

        self.dwe = DataWithErrors(self.data)


if __name__ == "__main__":
    unittest.main()