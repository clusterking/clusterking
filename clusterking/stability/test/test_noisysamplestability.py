#!/usr/bin/env python3

# std
import unittest
import tempfile

# ours
from clusterking.stability.noisysamplestability import (
    NoisySample,
    NoisySampleResult,
    NoisySampleStabilityTester,
)
from clusterking.scan.scanner import Scanner
from clusterking.data.data import Data
from clusterking.cluster import KmeansCluster


# noinspection PyUnusedLocal
def func_zero(coeffs):
    return 0.0


class TestNoisySample(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def cleanUp(self):
        self.tmpdir.cleanup()

    def test_noisy_sample(self):
        d = Data()
        s = Scanner()
        s.set_no_workers(1)
        s.set_spoints_equidist({"a": (0, 1, 2)})
        s.set_dfunction(func_zero)
        ns = NoisySample()
        ns.set_repeat(1)
        ns.set_noise("gauss", mean=0.0, sigma=1 / 30 / 4)
        nsr = ns.run(scanner=s, data=d)
        self.assertEqual(len(nsr.samples), 2)
        nsr.write(self.tmpdir.name, non_empty="raise")
        nsr_loaded = NoisySampleResult.load(self.tmpdir.name)
        for i in range(2):
            self.assertDictEqual(
                nsr.samples[i].df.to_dict(), nsr_loaded.samples[i].df.to_dict()
            )

        c = KmeansCluster()
        c.set_kmeans_options(n_clusters=2)
        nsst = NoisySampleStabilityTester()
        nsst.run(nsr, cluster=c)
