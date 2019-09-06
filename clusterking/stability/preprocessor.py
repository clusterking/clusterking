#!/usr/bin/env python3

# std
from abc import abstractmethod
from typing import Dict

# 3rd
import numpy as np

# ours
from clusterking.result import AbstractResult
from clusterking.worker import AbstractWorker
from clusterking.data.data import Data


class PreprocessorResult(AbstractResult):
    def __init__(self, data1, data2):
        super().__init__()
        self.data1 = data1
        self.data2 = data2


class Preprocessor(AbstractWorker):
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

    def run(self, data1: Data, data2: Data) -> PreprocessorResult:
        """ Run.

        Args:
            data1: "original" :class:`~clusterking.data.data.Data` object
            data2: "other" :class:`~clusterking.data.data.Data` object

        Returns:
            :class:`~PreprocessorResult`
        """
        return PreprocessorResult(data1=data1, data2=data2)


class ClusterMatcherResult(PreprocessorResult):
    def __init__(self, data1, data2, rename_dct):
        super().__init__(data1=data1, data2=data2)
        self.rename_dct = rename_dct


class ClusterMatcher(Preprocessor):
    """ Cluster names are arbitrary in general, i.e. when trying to compare
    two clustered datasets and trying to calculate a figure of merit, we have
    to match the names together.
    This is donen by this worker class.
    """

    def __init__(self, *args, cluster_column="cluster", **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_column = cluster_column

    @abstractmethod
    def run(self, data1: Data, data2: Data) -> ClusterMatcherResult:
        """

        Args:
            data1: "original" :class:`~clusterking.data.data.Data` object
            data2: "other" :class:`~clusterking.data.data.Data` object

        Returns:
            :class:`~ClusterMatcherResult`
        """
        pass


class TrivialClusterMatcher(ClusterMatcher):
    """ Thus subclass of :class:`CCMatcher` maps cluster names from the
    first clustering to the cluster name of the second that maximizes
    the number of sample points that lie in the same cluster.
    It also only returns the intersection of the indizes of both Series.
    """

    def run(self, data1: Data, data2: Data) -> ClusterMatcherResult:
        # todo [perf, low effort, med prio]: for speedup: only use pd.Series of
        #   clusters
        ndata1 = data1.copy(deep=True)
        ndata2 = data2.copy(deep=True)

        # 1. Throw out
        index_intersection = set(ndata1.df.index).intersection(
            set(ndata2.df.index)
        )
        ndata1.df = ndata1.df.loc[index_intersection]
        ndata2.df = ndata2.df.loc[index_intersection]

        # 2. Rename clusters
        clusters2 = set(ndata2.df[self.cluster_column])
        dct = {}
        for cluster2 in clusters2:
            mask = ndata2.df[self.cluster_column] == cluster2
            most_likely = np.argmax(
                np.bincount(ndata1.df[self.cluster_column][mask])
            )
            dct[cluster2] = most_likely

        ndata2.df[self.cluster_column] = ndata2.df[self.cluster_column].map(dct)

        return ClusterMatcherResult(data1=ndata1, data2=ndata2, rename_dct=dct)


class FirstComeFirstServe1DClusterMatcher(ClusterMatcher):
    """ This subclass of :class:`CCMatcher` works only for 1D parameter spaces.
    It simply sorts the first points of each cluster and enumerates them
    in order to get a unique name for each cluster."""

    def run(self, data1: Data, data2: Data) -> ClusterMatcherResult:
        ndata1 = data1.copy(deep=True)
        ndata2 = data2.copy(deep=True)

        nclusters1 = len(data1.df[self.cluster_column].unique())
        nclusters2 = len(data2.df[self.cluster_column].unique())
        if nclusters1 != nclusters2:
            raise ValueError("Cluster numbers don't match")
        order1 = self._get_order_of_clusters(data1)
        order2 = self._get_order_of_clusters(data2)
        order1_inverted = {value: key for key, value in order1.items()}
        rename_dct = {}
        for cluster in order2:
            rename_dct[cluster] = order1_inverted[order2[cluster]]

        ndata2.df[self.cluster_column] = ndata2.df[self.cluster_column].map(
            rename_dct
        )
        return ClusterMatcherResult(
            data1=ndata1, data2=ndata2, rename_dct=rename_dct
        )

    def _get_order_of_clusters(self, data: Data) -> Dict[int, int]:
        cluster2min = {}
        uclusters = data.df[self.cluster_column].unique()
        for ucluster in uclusters:
            cluster2min[ucluster] = data.df[
                data.df[self.cluster_column] == ucluster
            ][data.par_cols[0]].min()
        sorted_mins = sorted(list(cluster2min.values()))
        return {
            ucluster: sorted_mins.index(cluster2min[ucluster])
            for ucluster in uclusters
        }
