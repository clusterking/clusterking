#!/usr/bin/env python3

# std
import unittest
from pathlib import Path

# ours
from clusterking.cluster.kmeans_cluster import KmeansCluster
from clusterking.data import Data


class TestKmeansCluster(unittest.TestCase):
    def setUp(self):
        self.ddir = Path(__file__).parent / "data"
        self.dname = "1d.sql"
        self.d = Data(self.ddir / self.dname)

    def test_with_ncluster(self):
        c = KmeansCluster()
        c.set_kmeans_options(n_clusters=3)
        r = c.run(self.d)
        r.write()
        self.assertEqual(len(self.d.clusters()), 3)


if __name__ == "__main__":
    unittest.main()
