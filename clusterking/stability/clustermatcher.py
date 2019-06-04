#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod

# 3rd
import numpy as np
import pandas as pd


class ClusterMatcher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, data1, data2):
        pass


class TrivialClusterMatcher(ClusterMatcher):
    def run(self, clustered1: pd.Series, clustered2: pd.Series):
        clusters1 = set(clustered1)
        dct = {}
        for cluster1 in clusters1:
            mask = clustered1 == cluster1
            most_likely = np.argmax(np.bincount(clustered2[mask]))
            dct[cluster1] = most_likely
        return dct
