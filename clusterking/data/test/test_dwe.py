#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# 3rd
import numpy as np

# ours
from clusterking.util.testing import MyTestCase
from clusterking.data.dwe import DataWithErrors


class TestDataWithErrors(MyTestCase):
    def setUp(self):
        dpath = Path(__file__).parent / "data" / "test.sql"
        self.data = [[100.0, 200.0], [400.0, 500.0]]
        self.dwe = DataWithErrors(dpath)

    def ndwe(self):
        return self.dwe.copy(deep=True)

    def test_empty(self):
        dwe = self.ndwe()
        self.assertEqual(dwe.abs_cov.shape, (2, 2))
        self.assertAllClose(dwe.rel_cov, np.zeros((2, 2)))
        self.assertFalse(dwe.poisson_errors)

    def test_data_no_errors(self):
        dwe = self.ndwe()
        self.assertAllClose(dwe.data(), self.data)
        self.assertAllClose(
            dwe.data(normalize=True), [[1 / 3, 2 / 3], [4 / 9, 5 / 9]]
        )
        all_zero = np.zeros((2, 2))
        unit = np.eye(2)
        self.assertAllClose(dwe.rel_cov, all_zero)
        self.assertAllClose(dwe.abs_cov, all_zero)
        self.assertAllClose(dwe.cov(), all_zero)
        self.assertAllClose(dwe.corr(), unit)
        self.assertAllClose(dwe.data(decorrelate=True), self.data)

    # -------------------------------------------------------------------------

    def test_reset_errors(self):
        dwe = self.ndwe()
        cov = [[4.0, 4.0], [4.0, 16.0]]
        dwe.add_err_cov(cov)
        dwe.add_err_corr(1, np.identity(2))
        dwe.add_err_uncorr(0.3)
        dwe.add_err_poisson(normalization_scale=25)
        dwe.reset_errors()
        self.assertEqual(np.count_nonzero(dwe.cov()), 0)
        self.assertEqual(np.count_nonzero(dwe.abs_cov), 0)
        self.assertEqual(np.count_nonzero(dwe.rel_cov), 0)
        self.assertFalse(dwe.poisson_errors)
        self.assertEqual(dwe.poisson_errors_scale, 1.0)
        self.assertAllClose(
            dwe.corr(), np.tile(np.eye(dwe.nbins), (dwe.n, 1, 1))
        )

    def test_add_err_cov(self):
        dwe = self.ndwe()
        # Equal for all data points
        cov = [[4.0, 4.0], [4.0, 16.0]]
        dwe.add_err_cov(cov)
        self.assertAllClose(dwe.cov(), cov)
        self.assertAllClose(dwe.corr(), [[1.0, 1 / 2], [1 / 2, 1.0]])
        self.assertAllClose(dwe.err(), [2.0, 4.0])

    def test_add_err_corr(self):
        dwe = self.ndwe()
        dwe.add_err_corr(1, np.identity(2))
        self.assertAllClose(dwe.corr(), np.identity(2))

        corr = [[1.0, 0.32], [0.4, 1.0]]
        dwe = self.ndwe()
        dwe.add_err_corr(1.0, corr)
        self.assertAllClose(dwe.corr(), corr)
        self.assertAllClose(dwe.err(), 1)

        dwe = self.ndwe()
        err = [1.52, 2.34]
        dwe.add_err_corr(err, corr)
        self.assertAllClose(dwe.err(), err)
        self.assertAllClose(dwe.corr(), corr)

        dwe.add_err_corr(err, corr)
        self.assertAllClose(dwe.corr(), corr)

    def test_add_err_uncorr(self):
        dwe = self.ndwe()
        dwe.add_err_uncorr(0.3)
        self.assertAllClose(dwe.corr(), np.identity(2))
        self.assertAllClose(dwe.err(), 0.3)

        dwe = self.ndwe()
        err = [0.3, 1.5]
        dwe.add_err_uncorr(err)
        self.assertAllClose(dwe.corr(), np.identity(2))
        self.assertAllClose(dwe.err(), err)

    def test_add_err_maxcorr(self):
        dwe = self.ndwe()
        dwe.add_err_maxcorr(0.3)
        self.assertAllClose(dwe.corr(), np.ones((2, 2, 2)))
        self.assertAllClose(dwe.err(), 0.3)

        dwe = self.ndwe()
        err = [0.3, 1.5]
        dwe.add_err_maxcorr(err)
        self.assertAllClose(dwe.corr(), np.ones((2, 2, 2)))
        self.assertAllClose(dwe.err(), err)

    # todo: test rel_err

    # --------------------------------------------------------------------------

    def test_add_err_poisson(self):
        dwe = self.ndwe()
        dwe.add_err_poisson()
        self.assertAllClose(dwe.err(), np.sqrt(self.data))
        self.assertAllClose(dwe.err(relative=True), 1 / np.sqrt(self.data))
        self.assertAllClose(dwe.corr(), np.eye(len(self.data)))

    def test_add_err_poisson_scaled_relative(self):
        # Now we increase the statistics by a factor of 4 and expect that the
        # Normed errors are reduced by a factor of 2.
        dwe1 = self.ndwe()
        dwe1.add_err_poisson()
        rel_err1 = dwe1.err(relative=True)
        dwe2 = self.ndwe()
        dwe2.add_err_poisson(normalization_scale=4)
        rel_err2 = dwe2.err(relative=True)
        self.assertAllClose(rel_err1, rel_err2 * 2)

    # --------------------------------------------------------------------------
    def test_plot_dist_err(self):
        self.dwe.plot_dist_err()


if __name__ == "__main__":
    unittest.main()
