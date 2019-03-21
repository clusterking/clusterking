#!/usr/bin/env python3

# std
import unittest
from pathlib import Path

# 3rd
import numpy as np

# ours
from bclustering.util.testing import MyTestCase
from bclustering.data.dwe import DataWithErrors


class TestDataWithErrors(MyTestCase):
    def setUp(self):
        self.ddir = Path(__file__).parent / "data"
        self.dname = "test_scan"
        self.data = [[100, 200], [400, 500]]

    def test_data(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        self.assertAllClose(
            dwe.data(),
            self.data
        )
        self.assertAllClose(
            dwe.data(normalize=True),
            [
                [1/3, 2/3],
                [4/9, 5/9]
            ]
        )
        # fixme
        # self.assertAllClose(
        #     dwe.data(decorrelate=True),
        #     self.data
        # )

    # -------------------------------------------------------------------------

    def test_add_err_cov(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        # Equal for all data points
        cov = [[4, 4], [4, 16]]
        dwe.add_err_cov(cov)
        self.assertAllClose(
            dwe.cov(),
            cov
        )
        self.assertAllClose(
            dwe.corr(),
            [[1, 1/2], [1/2, 1]]
        )
        self.assertAllClose(
            dwe.err(),
            [2, 4]
        )
        # Different
        cov = [
            [[4, 4], [4, 16]],
            [[9, 0], [0, 1]]
        ]
        dwe = DataWithErrors(self.ddir, self.dname)
        dwe.add_err_cov(cov)
        self.assertAllClose(
            dwe.corr(),
            [
                [[1, 1/2], [1/2, 1]],
                [[1, 0], [0, 1]]
            ]
        )
        self.assertAllClose(
            dwe.err(),
            [
                [2, 4],
                [3, 1]
            ]
        )

    def test_add_err_corr(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        dwe.add_err_corr(1, np.identity(2))
        self.assertAllClose(
            dwe.corr(),
            np.identity(2)
        )

        corr = [
            [[1, 0.32], [0.4, 1]],
            [[1, 0.1], [0.5, 1]]
        ]
        dwe = DataWithErrors(self.ddir, self.dname)
        dwe.add_err_corr(1., corr)
        self.assertAllClose(
            dwe.corr(),
            corr
        )
        self.assertAllClose(
            dwe.err(),
            1
        )

        dwe = DataWithErrors(self.ddir, self.dname)
        err = [[1.52, 2.34], [3.87, 4.56]]
        dwe.add_err_corr(err, corr)
        self.assertAllClose(
            dwe.err(),
            err
        )
        self.assertAllClose(
            dwe.corr(),
            corr
        )

        dwe.add_err_corr(err, corr)
        self.assertAllClose(
            dwe.corr(),
            corr
        )

    def test_add_err_uncorr(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        dwe.add_err_uncorr(0.3)
        self.assertAllClose(
            dwe.corr(),
            np.identity(2)
        )
        self.assertAllClose(
            dwe.err(),
            0.3
        )

        dwe = DataWithErrors(self.ddir, self.dname)
        err = [0.3, 1.5]
        dwe.add_err_uncorr(err)
        self.assertAllClose(
            dwe.corr(),
            np.identity(2)
        )
        self.assertAllClose(
            dwe.err(),
            err
        )

    def test_add_err_maxcorr(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        dwe.add_err_maxcorr(0.3)
        self.assertAllClose(
            dwe.corr(),
            np.ones((2, 2, 2))
        )
        self.assertAllClose(
            dwe.err(),
            0.3
        )

        dwe = DataWithErrors(self.ddir, self.dname)
        err = [0.3, 1.5]
        dwe.add_err_maxcorr(err)
        self.assertAllClose(
            dwe.corr(),
            np.ones((2, 2, 2))
        )
        self.assertAllClose(
            dwe.err(),
            err
        )


if __name__ == "__main__":
    unittest.main()
