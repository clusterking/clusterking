#!/usr/bin/env python3

# std
import unittest

# ours
from clusterking.maths.statistics import (
    cov2corr,
    cov2err,
    rel2abs_cov,
    abs2rel_cov,
    corr2cov,
)
from clusterking.util.testing import MyTestCase


class TestStatistics(MyTestCase):
    """ To test the conversion functions cov2err etc.
    Subclassed for specific test cases """

    def setUp(self):
        self.data = {
            "1D": {
                "cov": [[4, 4], [-4, 16]],
                "cov_rel": [[1, 2], [-2, 16]],
                "err": [2, 4],
                "corr": [[1, 1 / 2], [-1 / 2, 1]],
                "data": [2, 1],
            },
            "2D": {
                "cov": [[[4, 4], [-4, 16]], [[4, 0], [0, 25]]],
                "cov_rel": [[[1, 2], [-2, 16]], [[4, 0], [0, 25]]],
                "err": [[2, 4], [2, 5]],
                "corr": [[[1, 1 / 2], [-1 / 2, 1]], [[1, 0], [0, 1]]],
                "data": [[2, 1], [1, 1]],
            },
        }

    def test_cov2err(self):
        for dataset, data in self.data.items():
            with self.subTest(dataset=dataset):
                self.assertAllClose(cov2err(data["cov"]), data["err"])

    def test_corr2cov(self):
        for dataset, data in self.data.items():
            with self.subTest(dataset=dataset):
                self.assertAllClose(
                    corr2cov(data["corr"], data["err"]), data["cov"]
                )

    def test_cov2corr(self):
        for dataset, data in self.data.items():
            with self.subTest(dataset=dataset):
                self.assertAllClose(cov2corr(data["cov"]), data["corr"])

    def test_rel2abs_cov(self):
        for dataset, data in self.data.items():
            with self.subTest(dataset=dataset):
                self.assertAllClose(
                    rel2abs_cov(data["cov_rel"], data["data"]), data["cov"]
                )

    def test_abs2rel_cov(self):
        for dataset, data in self.data.items():
            with self.subTest(dataset=dataset):
                self.assertAllClose(
                    abs2rel_cov(data["cov"], data["data"]), data["cov_rel"]
                )


if __name__ == "__main__":
    unittest.main()
