#!/usr/bin/env python3

# std
import unittest
import tempfile

# ours
from clusterking.stability.subsamplestability import SubSampleStabilityTester
from clusterking.scan.scanner import Scanner
from clusterking.data.data import Data
from clusterking.cluster import KmeansCluster


# noinspection PyUnusedLocal
def func_one(coeffs):
    return 1.0


class TestSubSampleStability(unittest.TestCase):
    def test_sss(self):
        d = Data()
        s = Scanner()
        s.set_no_workers(1)
        s.set_spoints_equidist({"a": (0, 1, 4)})
        s.set_dfunction(func_one)
        s.run(d).write()
        c = KmeansCluster()
        c.set_kmeans_options(n_clusters=2)
        ssst = SubSampleStabilityTester()
        ssst.set_sampling(frac=0.95)
        ssst.set_repeat(2)
        ssst.run(data=d, cluster=c)
