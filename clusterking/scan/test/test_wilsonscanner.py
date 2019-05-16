#!/usr/bin/env python3

# std
import unittest

# 3rd
import numpy as np

# ours
from clusterking.util.testing import MyTestCase
from clusterking.scan.wilsonscanner import WilsonScanner
from clusterking.data.data import Data


def simple_func(w, q):
    return q+1


class TestWilsonScannerRun(MyTestCase):
    def setUp(self):
        self.s = WilsonScanner(scale=5, eft='WET', basis='flavio')
        self.s.set_spoints_equidist(
            {
                "CVL_bctaunutau": (-1, 1, 2),
                "CSL_bctaunutau": (-1, 1, 2),
                "CT_bctaunutau": (-1, 1, 2)
            },
        )
        self.s.set_dfunction(
            simple_func,
            binning=[0, 1, 2],
            normalize=True
        )
        self.d = Data()

    def test_run(self):
        self.s.run(self.d)
        self.assertEqual(self.d.n, 8)
        self.assertEqual(self.d.nbins, 2)
        self.assertEqual(self.d.npars, 3)


class TestWilsonScanner(MyTestCase):

    def test_spoints_equidist(self):
        s = WilsonScanner(scale=5, eft='WET', basis='flavio')
        s.set_spoints_equidist(
            {
                "CVL_bctaunutau": (-1, 1, 2),
                "CSL_bctaunutau": (-1, 1, 3),
                "CT_bctaunutau": (-1, 1, 4)
            },
        )
        self.assertEqual(
            len(s.spoints), 2*3*4
        )

    def test_spoints_equidist_complex(self):
        s = WilsonScanner(scale=5, eft='WET', basis='flavio')
        s.set_spoints_equidist(
            {
                "CVL_bctaunutau": (0, 1, 2),
                "im_CVL_bctaunutau": (0, 1, 2),
            },
        )
        self.assertEqual(
            len(s.spoints), 2*2
        )
        self.assertAllClose(
            s.spoints,
            np.array([[0.], [1.j], [1.], [1.+1.j]])
        )

    def test_properties(self):
        s = WilsonScanner(scale=5, eft='WET', basis='flavio')
        self.assertEqual(s.scale, 5)
        self.assertEqual(s.eft, "WET")
        self.assertEqual(s.basis, "flavio")


if __name__ == "__main__":
    unittest.main()