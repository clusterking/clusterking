#!/usr/bin/env python3

# std
import unittest

# 3rd
import pandas as pd

# ours
from clusterking.stability.preprocessor import TrivialClusterMatcher
from clusterking.data.data import Data


class TestTrivialClusterMatcher(unittest.TestCase):
    def test(self):
        d1 = Data()
        d2 = Data()
        d1.df = pd.DataFrame({"cluster": [1, 1, 2, 2, 3]})
        d2.df = pd.DataFrame({"cluster": [2, 2, 3, 3, 1]})
        ttcmr = TrivialClusterMatcher().run(d1, d2)
        self.assertDictEqual(ttcmr.rename_dct, {2: 1, 3: 2, 1: 3})


if __name__ == "__main__":
    unittest.main()
