#!/usr/bin/env python3

# std
from pathlib import Path
import unittest

# ours
from clusterking.util.testing import MyTestCase
from clusterking.data.data import Data
from clusterking.cluster.hierarchy_cluster import HierarchyCluster


class TestHierarchyCluster(MyTestCase):
    def setUp(self):
        self.ddir = Path(__file__).parent / "data"
        self.dname = "1d.sql"
        self.d = Data(self.ddir / self.dname)

    def test_cluster(self):
        c = HierarchyCluster(self.d)
        c.set_metric()
        c.build_hierarchy()
        c.cluster(max_d=0.75)
        c.write()
        c.cluster(max_d=1.5)
        c.write(cluster_column="cluster15")
        # The minimal distance between our distributions is 1, so they all
        # end up in different clusters
        self.assertEqual(len(self.d.clusters()), self.d.n)
        # This is a bit unfortunate, since we have so many distribution pairs
        # with equal distance (so it's up to the implementation of the algorithm
        # , which clusters develop) but this is what happened so far:
        self.assertEqual(len(self.d.clusters(cluster_column="cluster15")), 6)

    def test_dendrogram_plot(self):
        c = HierarchyCluster(self.d)
        c.set_metric()
        c.build_hierarchy()
        c.dendrogram()


if __name__ == "__main__":
    unittest.main()