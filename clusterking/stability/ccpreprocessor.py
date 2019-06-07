#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod
import copy

# 3rd
import numpy as np
import pandas as pd

# ours


class CCPreprocessorResult(object):
    def __init__(self, clustered1, clustered2):
        self.clustered1 = clustered1
        self.clustered2 = clustered2


class CCPreprocessor(object):
    """ Cluster names are arbitrary in general, i.e. when trying to compare
    two clustered datasets and trying to calculate a figure of merit, we have
    to match the names together.
    This is donen by this worker class.
    """

    def __init__(self, name=None):
        self._name = None

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
        clustered1 = clustered1[index_intersection]
        clustered2 = clustered2[index_intersection]

        # 2. Rename clusters
        clusters1 = set(clustered1)
        dct = {}
        for cluster1 in clusters1:
            mask = clustered1 == cluster1
            most_likely = np.argmax(np.bincount(clustered2[mask]))
            dct[cluster1] = most_likely

        return CCPreprocessorResult(
            clustered1=clustered1, clustered2=clustered2
        )
