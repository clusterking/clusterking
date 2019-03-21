#!/usr/bin/env python3

# std
import unittest

# ours
from bclustering.maths.statistics import *
from bclustering.util.testing import MyTestCase


class TestStatistics(MyTestCase):
    """ To test the conversion functions cov2err etc.
    Subclassed for specific test cases """

    def __init__( self, *args, **kwargs ):
        super(TestStatistics, self).__init__(*args, **kwargs)
        if self.__class__ == TestStatistics:
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


class TestStatistics1D(TestStatistics):
    """ Test metric utils with 1 datapoint.
    Testing methods inherited from TestStatistics
    """
    def setUp(self):
        self.cov = [[4, 4], [-4, 16]]
        self.cov_rel = [[1, 2], [-2, 16]]
        self.err = [2, 4]
        self.corr = [[1, 1/2], [-1/2, 1]]
        self.data = [2, 1]


class TestStatistics2D(TestStatistics):
    """ Test metric utils with several datapoints
    Testing methods inherited from TestStatistics
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


if __name__ == "__main__":
    unittest.main()