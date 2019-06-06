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
        d = self.d.copy()
        c = HierarchyCluster()
        c.set_metric("euclidean")
        c.set_max_d(0.75)
        c.run(d).write()
        c.set_max_d(1.5)
        c.run(d).write(cluster_column="cluster15")
        # The minimal distance between our distributions is 1, so they all
        # end up in different clusters
        self.assertEqual(len(d.clusters()), self.d.n)
        # This is a bit unfortunate, since we have so many distribution pairs
        # with equal distance (so it's up to the implementation of the algorithm
        # , which clusters develop) but this is what happened so far:
        self.assertEqual(len(d.clusters(cluster_column="cluster15")), 6)

    def test_reuse_hierarchy(self):
        d = self.d.copy()
        c = HierarchyCluster()
        c.set_metric("euclidean")
        c.set_max_d(1.5)
        r = c.run(d)
        r.write()
        r2 = c.run(d, reuse_hierarchy_from=r)
        r2.write(cluster_column="reused")
        self.assertListEqual(d.df["cluster"].tolist(), d.df["reused"].tolist())

    def test_reuse_hierarchy_fail_different_data(self):
        d = self.d.copy()
        e = self.d.copy()
        c = HierarchyCluster()
        c.set_metric("euclidean")
        c.set_max_d(1.5)
        r = c.run(d)
        r.write()
        with self.assertRaises(ValueError) as ex:
            c.run(e, reuse_hierarchy_from=r)
        self.assertTrue("different data object" in str(ex.exception))

    def test_reuse_hierarchy_fail_different_cluster(self):
        d = self.d.copy()
        c = HierarchyCluster()
        c2 = HierarchyCluster()
        c.set_metric("euclidean")
        c.set_max_d(1.5)
        c2.set_metric("euclidean")
        c2.set_max_d(1.5)
        r = c.run(d)
        r.write()
        with self.assertRaises(ValueError) as e:
            c2.run(e, reuse_hierarchy_from=r)
        self.assertTrue("different HierarchyCluster object" in str(e.exception))

    def test_hierarchy_cluster_no_max_d(self):
        d = self.d.copy()
        c = HierarchyCluster()
        with self.assertRaises(ValueError) as e:
            c.run(d)
        self.assertTrue("set_max_d" in str(e.exception))

    def test_dendrogram_plot(self):
        c = HierarchyCluster()
        c.set_metric()
        c.set_max_d(0.2)
        r = c.run(self.d)
        r.dendrogram()


if __name__ == "__main__":
    unittest.main()
