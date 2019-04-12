#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# ours
from clusterking.util.log import silence_all_logs
from clusterking.util.testing import MyTestCase
from clusterking.data.data import Data


class TestData(MyTestCase):
    def setUp(self):
        silence_all_logs()
        self.ddir = Path(__file__).parent / "data"
        self.dname = "test"
        self.data = [[100, 200], [400, 500]]
        self.d = Data(self.ddir, self.dname)

    # **************************************************************************
    # Property shortcuts
    # **************************************************************************

    def test_bin_cols(self):
        self.assertEqual(self.d.bin_cols, ["bin0", "bin1"])

    def test_par_cols(self):
        self.assertEqual(
            self.d.par_cols,
            ["CVL_bctaunutau", "CT_bctaunutau", "CSL_bctaunutau"]
        )

    def test_n(self):
        self.assertEqual(self.d.n, 2)

    def test_nbins(self):
        self.assertEqual(self.d.nbins, 2)

    def test_npars(self):
        self.assertEqual(self.d.npars, 3)

    # **************************************************************************
    # Returning things
    # **************************************************************************

    def test_data(self):
        self.assertAllClose(
            self.d.data(),
            self.data
        )

    def test_norms(self):
        self.assertAllClose(
            self.d.norms(),
            [300, 900]
        )

    def test_clusters(self):
        self.assertEqual(
            self.d.clusters(),
            [0]
        )
        self.assertEqual(
            self.d.clusters(cluster_column="other_cluster"),
            [0, 1]
        )

    def test_get_param_values(self):
        self.assertEqual(
            sorted(list(self.d.get_param_values().keys())),
            sorted(["CVL_bctaunutau", "CT_bctaunutau", "CSL_bctaunutau"])
        )
        self.assertAlmostEqual(
            self.d.get_param_values("CVL_bctaunutau")[0],
            -1.
        )
        self.assertAlmostEqual(
            self.d.get_param_values("CT_bctaunutau")[1],
            0.
        )

    def test_data_normed(self):
        self.assertAllClose(
            self.d.data(normalize=True),
            [[1/3, 2/3], [4/9, 5/9]]
        )

    # **************************************************************************
    # Subsample
    # **************************************************************************

    # see next class


class TestSubSample(MyTestCase):
    def setUp(self):
        silence_all_logs()
        self.ddir = Path(__file__).parent / "data"
        self.dname = "test_longer"
        self.d = Data(self.ddir, self.dname)

    def test_only_bpoints(self):
        self.assertEqual(self.d.only_bpoints().n, 1)
        self.assertEqual(
            self.d.only_bpoints(bpoint_column="bpoint1").n,
            2
        )
        self.assertEqual(
            self.d.only_bpoints(bpoint_column="bpoint2").n,
            3
        )

    def test_fix_param(self):
        e = self.d.fix_param(a=0)
        self.assertEqual(e.n, 16)
        self.assertAllClose(e.get_param_values("a"), [0.])
        e = self.d.fix_param(a=-100)
        self.assertEqual(e.n, 16)
        self.assertAllClose(e.get_param_values("a"), [0.])

    # def test_fix_param_bpoints(self):
    #     e = self.d.fix_param(CT_bctaunutau=[], bpoints=True)
    #     self.assertEqual(e.n, 1)
    #     e = self.d.fix_param(CT_bctaunutau=[], bpoint_slices=True)
    #     self.assertEqual(e.n, 1)
    #     e = self.d.fix_param(
    #         CT_bctaunutau=[],
    #         bpoints=True,
    #         bpoint_column="other_bpoint"
    #     )
    #     self.assertEqual(e.n, 2)


if __name__ == "__main__":
    unittest.main()
