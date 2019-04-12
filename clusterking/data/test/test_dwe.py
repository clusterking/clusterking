#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# 3rd
import numpy as np

# ours
from clusterking.util.log import silence_all_logs
from clusterking.util.testing import MyTestCase
from clusterking.data.dwe import DataWithErrors


class TestDataWithErrors(MyTestCase):
    def setUp(self):
        silence_all_logs()
        self.ddir = Path(__file__).parent / "data"
        self.dname = "test"
        self.data = [[100., 200.], [400., 500.]]

    def test_empty(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        self.assertEqual(
            dwe.abs_cov.shape,
            (2, 2)
        )
        self.assertAllClose(
            dwe.rel_cov,
            np.zeros((2, 2))
        )
        self.assertFalse(dwe.poisson_errors)

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
        cov = [[4., 4.], [4., 16.]]
        dwe.add_err_cov(cov)
        self.assertAllClose(
            dwe.cov(),
            cov
        )
        self.assertAllClose(
            dwe.corr(),
            [[1., 1/2], [1/2, 1.]]
        )
        self.assertAllClose(
            dwe.err(),
            [2., 4.]
        )

    def test_add_err_corr(self):
        dwe = DataWithErrors(self.ddir, self.dname)
        dwe.add_err_corr(1, np.identity(2))
        self.assertAllClose(
            dwe.corr(),
            np.identity(2)
        )

        corr = [
            [1., 0.32],
            [0.4, 1.],
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
        err = [1.52, 2.34]
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

    # todo: test rel_err


if __name__ == "__main__":
    unittest.main()
