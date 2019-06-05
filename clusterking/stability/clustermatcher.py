#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod

# 3rd
import numpy as np
import pandas as pd


class ClusterMatcherResult(object):
    """ Result of :class:`ClusterMatcher`. """

    def __init__(self, dct: dict):
        #: Dictionary cluster name first clustering -> cluster name second
        #: clustering.
        self.dct = dct  # type: dict


class ClusterMatcher(ABC):
    """ Cluster names are arbitrary in general, i.e. when trying to compare
    two clustered datasets and trying to calculate a figure of merit, we have
    to match the names together.
    This is donen by this worker class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(
        self, clustered1: pd.Series, clustered2: pd.Series
    ) -> ClusterMatcherResult:
        """ Run.

        Args:
            clustered1: :class:`pandas.Series` of cluster numbers
            clustered2: :class:`pandas.Series` of cluster numbers

        Returns:

        """
        pass


class TrivialClusterMatcherResult(ClusterMatcherResult):
    pass


class TrivialClusterMatcher(ClusterMatcher):
    """ Thus subclass of :class:`ClusterMatcher` maps cluster names from the
    first clustering to the cluster name of the second that maximizes
    the number of sample points that lie in the same cluster.
    """

    def run(self, clustered1: pd.Series, clustered2: pd.Series):
        clusters1 = set(clustered1)
        dct = {}
        for cluster1 in clusters1:
            mask = clustered1 == cluster1
            most_likely = np.argmax(np.bincount(clustered2[mask]))
            dct[cluster1] = most_likely
        return TrivialClusterMatcherResult(dct=dct)
