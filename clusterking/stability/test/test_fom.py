#!/usr/bin/env python3

# std
import unittest

# 3rd
import pandas as pd

# ours
from clusterking.data.data import Data
from clusterking.stability.fom import MatchingClusters, DeltaNClusters
from clusterking.stability.preprocessor import TrivialClusterMatcher


class TestFOMs(unittest.TestCase):
    def setUp(self):
        self.d1 = Data()
        self.d2 = Data()
        self.d3 = Data()
        self.d4 = Data()
        self.d1.df = pd.DataFrame({"cluster": [1, 1, 2, 2, 3]})
        self.d2.df = pd.DataFrame({"cluster": [2, 2, 3, 3, 1]})
        self.d3.df = pd.DataFrame({"cluster": [2, 1, 2, 2, 3]})
        self.d4.df = pd.DataFrame({"cluster": [4, 1, 2, 2, 3]})

    def test_deltanclusters(self):
        fom = DeltaNClusters()
        self.assertEqual(fom.run(self.d1, self.d2).fom, 0)
        self.assertEqual(fom.run(self.d1, self.d3).fom, 0)
        self.assertEqual(fom.run(self.d1, self.d4).fom, -1)

    def test_matchingclusters(self):
        fom = MatchingClusters()
        self.assertAlmostEqual(fom.run(self.d1, self.d1).fom, 1.0)
        self.assertAlmostEqual(fom.run(self.d1, self.d2).fom, 0.0)
        self.assertAlmostEqual(fom.run(self.d1, self.d3).fom, 4 / 5)
        self.assertAlmostEqual(fom.run(self.d1, self.d4).fom, 4 / 5)

    def test_matchingclusters_with_preprocessor(self):
        fom = MatchingClusters(preprocessor=TrivialClusterMatcher())
        self.assertAlmostEqual(fom.run(self.d1, self.d1).fom, 1.0)
        self.assertAlmostEqual(fom.run(self.d1, self.d2).fom, 1.0)
