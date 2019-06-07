#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# ours
from clusterking.util.testing import MyTestCase
from clusterking.data.data import Data


class TestData(MyTestCase):
    def setUp(self):
        path = Path(__file__).parent / "data" / "test.sql"
        self.data = [[100, 200], [400, 500]]
        self.d = Data(path)

    def nd(self):
        return self.d.copy(deep=True)

    # **************************************************************************
    # Property shortcuts
    # **************************************************************************

    def test_bin_cols(self):
        self.assertEqual(self.d.bin_cols, ["bin0", "bin1"])

    def test_par_cols(self):
        self.assertEqual(
            self.d.par_cols,
            ["CVL_bctaunutau", "CT_bctaunutau", "CSL_bctaunutau"],
        )

    def test_n(self):
        self.assertEqual(self.d.n, 2)

    def test_nbins(self):
        self.assertEqual(self.d.nbins, 2)

    def test_npars(self):
        self.assertEqual(self.d.npars, 3)

    def test__dist_xrange(self):
        self.assertEqual(self.d._dist_xrange, (0, 20))

    # **************************************************************************
    # Returning things
    # **************************************************************************

    def test_data(self):
        self.assertAllClose(self.d.data(), self.data)

    def test_norms(self):
        self.assertAllClose(self.d.norms(), [300, 900])

    def test_clusters(self):
        self.assertEqual(self.d.clusters(), [0])
        self.assertEqual(
            self.d.clusters(cluster_column="other_cluster"), [0, 1]
        )

    def test_get_param_values(self):
        self.assertEqual(
            sorted(list(self.d.get_param_values().keys())),
            sorted(["CVL_bctaunutau", "CT_bctaunutau", "CSL_bctaunutau"]),
        )
        self.assertAlmostEqual(
            self.d.get_param_values("CVL_bctaunutau")[0], -1.0
        )
        self.assertAlmostEqual(self.d.get_param_values("CT_bctaunutau")[1], 0.0)

    def test_data_normed(self):
        self.assertAllClose(
            self.d.data(normalize=True), [[1 / 3, 2 / 3], [4 / 9, 5 / 9]]
        )

    # **************************************************************************
    # Subsample
    # **************************************************************************

    # see next class

    # **************************************************************************
    # Quick plots
    # **************************************************************************
    # We just check that they run without throwing.

    def test_plot_dist(self):
        self.d.plot_dist()

    def test_plot_dist_minmax(self):
        self.d.plot_dist_minmax()

    def test_plot_dist_box(self):
        self.d.plot_dist_box()

    def test_plot_clusters_scatter(self):
        self.d.plot_clusters_scatter(
            ["CVL_bctaunutau", "CT_bctaunutau", "CSL_bctaunutau"]
        )
        self.d.plot_clusters_scatter(["CVL_bctaunutau", "CT_bctaunutau"])

    def test_plot_clusters_fill(self):
        self.d.plot_clusters_fill(["CVL_bctaunutau", "CT_bctaunutau"])


class TestSubSample(MyTestCase):
    def setUp(self):
        path = Path(__file__).parent / "data" / "test_longer.sql"
        self.d = Data(path)

    def nd(self):
        return self.d.copy(deep=True)

    def test_only_bpoints(self):
        self.assertEqual(self.d.only_bpoints().n, 1)
        self.assertEqual(self.d.only_bpoints(bpoint_column="bpoint1").n, 2)
        self.assertEqual(self.d.only_bpoints(bpoint_column="bpoint2").n, 3)

    def test_fix_param(self):
        e = self.d.fix_param(a=0)
        self.assertEqual(e.n, 16)
        self.assertAllClose(e.get_param_values("a"), [0.0])

        e = self.d.fix_param(a=-100)
        self.assertEqual(e.n, 16)
        self.assertAllClose(e.get_param_values("a"), [0.0])

        e = self.d.fix_param(a=2.3)
        self.assertEqual(e.n, 16)
        self.assertAllClose(e.get_param_values("a"), [2.0])

        e = self.d.fix_param(a=[0, 2.3])
        self.assertEqual(e.n, 32)
        self.assertAllClose(e.get_param_values("a"), [0.0, 2.0])

        e = self.d.fix_param(a=[0, 2.3], b=0)
        self.assertEqual(e.n, 8)
        self.assertAllClose(e.get_param_values("a"), [0.0, 2.0])
        self.assertAllClose(e.get_param_values("b"), [0.0])

        e = self.d.fix_param(a=[0, 2.3], b=0, c=0.0)
        self.assertEqual(e.n, 2)
        self.assertAllClose(e.get_param_values("a"), [0.0, 2.0])
        self.assertAllClose(e.get_param_values("b"), [0.0])
        self.assertAllClose(e.get_param_values("c"), [0.0])

    def test_fix_param_bpoints(self):
        e = self.d.fix_param(a=[], bpoints=True)
        self.assertEqual(e.n, 1)

        e = self.d.fix_param(a=[], bpoints=True, bpoint_column="bpoint1")
        self.assertEqual(e.n, 2)

        e = self.d.fix_param(a=0.0, bpoints=True, bpoint_column="bpoint1")
        self.assertEqual(e.n, 16)

        e = self.d.fix_param(c=0.0, bpoints=True, bpoint_column="bpoint1")
        self.assertEqual(e.n, 17)

        e = self.d.fix_param(
            a=0.0, b=0.0, c=0.0, bpoints=True, bpoint_column="bpoint1"
        )
        self.assertEqual(e.n, 2)

    def test_fix_param_bpoint_slices(self):
        e = self.d.fix_param(a=[], bpoint_slices=True)
        self.assertEqual(e.n, 16)

        e = self.d.fix_param(c=[], bpoint_slices=True, bpoint_column="bpoint2")
        self.assertEqual(e.n, 3 * 16)

        e = self.d.fix_param(
            a=[], b=[], c=[], bpoint_slices=True, bpoint_column="bpoint2"
        )
        self.assertEqual(e.n, 3)

    def test_sample_param(self):
        e = self.d.sample_param(a=0)
        self.assertEqual(e.n, 0)

        e = self.d.sample_param(a=3)
        self.assertEqual(e.n, 3 * 4 * 4)

        e = self.d.sample_param(a=4)
        self.assertEqual(e.n, 4 * 4 * 4)

        e = self.d.sample_param(a=10)
        self.assertEqual(e.n, 4 * 4 * 4)

        e = self.d.sample_param(a=3, b=3, c=3)
        self.assertEqual(e.n, 3 * 3 * 3)

        e = self.d.sample_param(a=(0, 0.4, 3))
        self.assertEqual(e.n, 1 * 4 * 4)

        e = self.d.sample_param(a=(0, 1, 3))
        self.assertEqual(e.n, 2 * 4 * 4)

        e = self.d.sample_param(a=(0, 1, 3), b=2, c=2)
        self.assertEqual(e.n, 2 * 2 * 2)

        e = self.d.sample_param(a=(0, 1, 3), b=(0, 1, 3), c=2)
        self.assertEqual(e.n, 2 * 2 * 2)

    def test_sample_param_bpoints(self):
        e = self.d.sample_param(a=0, bpoints=True)
        self.assertEqual(e.n, 1)

        e = self.d.sample_param(a=0, bpoints=True, bpoint_column="bpoint2")
        self.assertEqual(e.n, 3)

    def test_sample_param_bpoint_slices(self):
        e = self.d.sample_param(a=0, bpoint_slices=True)
        self.assertEqual(e.n, 16)

        e = self.d.sample_param(
            a=0, bpoint_slices=True, bpoint_column="bpoint2"
        )
        self.assertEqual(e.n, 16)


if __name__ == "__main__":
    unittest.main()
