#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# 3rd

# ours
from clusterking.util.testing import MyTestCase
from clusterking.data.data import Data
from clusterking.benchmark.benchmark import Benchmark


class TestHierarchyCluster(MyTestCase):
    def setUp(self):
        self.ddir = Path(__file__).parent / "data"
        self.dname = "1d_clustered.sql"
        self.d = Data(self.ddir / self.dname)

    def test_cluster(self):
        b = Benchmark(self.d)
        b.set_metric()
        b.select_bpoints()
        b.write()
        # This is the cluster column where every spoint is its own cluster, so
        # all of them need to be benchmark points
        self.assertEqual(self.d.df["bpoint"].value_counts()[True], self.d.n)

        b = Benchmark(self.d, cluster_column="cluster1")
        b.set_metric()
        b.select_bpoints()
        b.write()
        # Only one cluster at all ==> only one bpoint
        self.assertEqual(self.d.df["bpoint"].value_counts()[True], 1)
        self.assertEqual(self.d.df[self.d.df["bpoint"]]["bin0"].values, 5)


if __name__ == "__main__":
    unittest.main()
