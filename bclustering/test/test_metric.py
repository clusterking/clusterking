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
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        self.assertTrue(np.allclose(
            a, b
        ))


class TestMetricUtils(MyTestCase):
    """ To test the conversion functions cov2err etc.
    Subclassed for specific test cases """

    def __init__( self, *args, **kwargs ):
        super(TestMetricUtils, self).__init__(*args, **kwargs)
        if self.__class__ == TestMetricUtils:
            # Make sure that unittest doesn't run the tests of this class,
            # only of its subclasses.
            self.run = lambda self, *args, **kwargs: None

    def setUp(self):
        self.cov = None
        self.cov_rel = None
        self.corr = None
        self.err = None
        self.data = None

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

    def test_rel2abs_cov(self):
        self.assertAllClose(
            rel2abs_cov(self.cov_rel, self.data),
            self.cov
        )

    def test_abs2rel_cov(self):
        self.assertAllClose(
            abs2rel_cov(self.cov, self.data),
            self.cov_rel
        )


class TestMetricUtils1D(TestMetricUtils):
    """ Test metric utils with 1 datapoint.
    Testing methods inherited from TestMetricUtils
    """
    def setUp(self):
        self.cov = [[4, 4], [-4, 16]]
        self.cov_rel = [[1, 2], [-2, 16]]
        self.err = [2, 4]
        self.corr = [[1, 1/2], [-1/2, 1]]
        self.data = [2, 1]


class TestMetricUtils2D(TestMetricUtils):
    """ Test metric utils with several datapoints
    Testing methods inherited from TestMetricUtils
    """
    def setUp(self):
        self.cov = [
            [[4, 4], [-4, 16]],
            [[4, 0], [0, 25]]
        ]
        self.cov_rel = [
            [[1, 2], [-2, 16]],
            [[4, 0], [0, 25]]
        ]
        self.err = [
            [2, 4],
            [2, 5]
        ]
        self.corr = [
            [[1, 1/2], [-1/2, 1]],
            [[1, 0], [0, 1]]
        ]
        self.data = [
            [2, 1],
            [1, 1]
        ]


class TestDataWithErrors(MyTestCase):
    def setUp(self):
        self.data = [
            [100, 200],
            [400, 500]
        ]

    def test_norms(self):
        self.assertAllClose(
            DataWithErrors(self.data).norms(),
            [300, 900]
        )

    def test_data(self):
        dwe = DataWithErrors(self.data)
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
        dwe = DataWithErrors(self.data)
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
        dwe = DataWithErrors(self.data)
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
        dwe = DataWithErrors(self.data)
        dwe.add_err_corr(1, np.identity(2))
        self.assertAllClose(
            dwe.corr(),
            np.identity(2)
        )

        corr = [
            [[1, 0.32], [0.4, 1]],
            [[1, 0.1], [0.5, 1]]
        ]
        dwe = DataWithErrors(self.data)
        dwe.add_err_corr(1., corr)
        self.assertAllClose(
            dwe.corr(),
            corr
        )
        self.assertAllClose(
            dwe.err(),
            1
        )

        dwe = DataWithErrors(self.data)
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
        dwe = DataWithErrors(self.data)
        dwe.add_err_uncorr(0.3)
        self.assertAllClose(
            dwe.corr(),
            np.identity(2)
        )
        self.assertAllClose(
            dwe.err(),
            0.3
        )

        dwe = DataWithErrors(self.data)
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
        dwe = DataWithErrors(self.data)
        dwe.add_err_maxcorr(0.3)
        self.assertAllClose(
            dwe.corr(),
            np.ones((2, 2, 2))
        )
        self.assertAllClose(
            dwe.err(),
            0.3
        )

        dwe = DataWithErrors(self.data)
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