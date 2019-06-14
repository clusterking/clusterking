#!/usr/bin/env python3

# std

# 3rd
import numpy as np
import pandas as pd

# ours
from clusterking.result import AbstractResult
from clusterking.worker import AbstractWorker


class CCPreprocessorResult(AbstractResult):
    def __init__(self, clustered1, clustered2):
        super().__init__()
        self.clustered1 = clustered1
        self.clustered2 = clustered2


class CCPreprocessor(AbstractWorker):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def name(self):
        if self._name is None:
            return str(type(self).__name__)
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def run(
        self, clustered1: pd.Series, clustered2: pd.Series
    ) -> CCPreprocessorResult:
        """ Run.

        Args:
            clustered1: :class:`pandas.Series` of cluster numbers
            clustered2: :class:`pandas.Series` of cluster numbers

        Returns:

        """
        return CCPreprocessorResult(
            clustered1=clustered1, clustered2=clustered2
        )


class ClusterMatcherResult(CCPreprocessorResult):
    def __init__(self, clustered1, clustered2, rename_dct):
        super().__init__(clustered1=clustered1, clustered2=clustered2)
        self.rename_dct = rename_dct


class ClusterMatcher(CCPreprocessor):
    """ Cluster names are arbitrary in general, i.e. when trying to compare
    two clustered datasets and trying to calculate a figure of merit, we have
    to match the names together.
    This is donen by this worker class.
    """


class TrivialClusterMatcher(CCPreprocessor):
    """ Thus subclass of :class:`CCMatcher` maps cluster names from the
    first clustering to the cluster name of the second that maximizes
    the number of sample points that lie in the same cluster.
    It also only returns the intersection of the indizes of both Series.
    """

    def run(self, clustered1: pd.Series, clustered2: pd.Series):
        # 1. Throw out
        index_intersection = set(clustered1.index).intersection(
            set(clustered2.index)
        )
        clustered1 = clustered1.loc[index_intersection]
        clustered2 = clustered2.loc[index_intersection]

        # 2. Rename clusters
        clusters2 = set(clustered2)
        dct = {}
        for cluster2 in clusters2:
            mask = clustered2 == cluster2
            most_likely = np.argmax(np.bincount(clustered1[mask]))
            dct[cluster2] = most_likely
        clustered2 = clustered2.map(dct)
        return ClusterMatcherResult(
            clustered1=clustered1, clustered2=clustered2, rename_dct=dct
        )
