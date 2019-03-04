#!/usr/bin/env python3

# std
import unittest

# 3rd
import numpy as np

# ours
from bclustering.metric import *


class TestMetricUtils1D(unittest.TestCase):
    def setUp(self):
        self.cov = np.array([[4, 4], [-4, 16]])
        self.err = np.array([2, 4])
        self.corr = np.array([[1, 1/2], [-1/2, 1]])

    def assertAllClose(self, a, b):
        self.assertTrue(np.allclose(
            a, b
        ))

    def test_cov2err(self):
        self.assertAllClose(
            cov2err(self.cov),
            self.err
        )

    def test_corr2cov(self):
        self.assertAllClose(
            corr2cov(self.corr, self.err),
            self.cov
        )

    def test_cov2corr(self):
        self.assertAllClose(
            cov2corr(self.cov),
            self.corr
        )


class TestMetricUtils2D(unittest.TestCase):
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

    def assertAllClose(self, a, b):
        self.assertTrue(np.allclose(
            a, b
        ))

    def test_cov2err(self):
        self.assertAllClose(
            cov2err(self.cov),
            self.err
        )

    def test_corr2cov(self):
        self.assertAllClose(
            corr2cov(self.corr, self.err),
            self.cov
        )

    def test_cov2corr(self):
        self.assertAllClose(
            cov2corr(self.cov),
            self.corr
        )


class TestDataWithErrors(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [100, 200],
            [400, 500]
        ])

        self.dwe = DataWithErrors(self.data)



if __name__ == "__main__":
    unittest.main()